import os
from contextlib import contextmanager

import mysql.connector
import streamlit as st


def _setting(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
        if "mysql" in st.secrets and name in st.secrets["mysql"]:
            return st.secrets["mysql"][name]
    except Exception:
        pass
    return os.getenv(name.upper(), default)


def get_db_config():
    return {
        "host": _setting("db_host", "localhost"),
        "port": int(_setting("db_port", 3306)),
        "user": _setting("db_user", "root"),
        "password": _setting("db_password", ""),
        "database": _setting("db_name", "brain_tumor_app"),
    }


@contextmanager
def get_connection():
    conn = mysql.connector.connect(**get_db_config())
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    config = get_db_config()
    database = config.pop("database")

    bootstrap_conn = mysql.connector.connect(**config)
    try:
        cursor = bootstrap_conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        bootstrap_conn.commit()
    finally:
        cursor.close()
        bootstrap_conn.close()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(150) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                file_name VARCHAR(255),
                predicted_class VARCHAR(50),
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                token_hash VARCHAR(64) NOT NULL UNIQUE,
                expires_at DATETIME NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()
        cursor.close()


def save_scan_record(user_id, file_name, predicted_class, confidence):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO scan_records (user_id, file_name, predicted_class, confidence)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, file_name, predicted_class, confidence),
        )
        conn.commit()
        cursor.close()


def get_scan_history(user_id, limit=20):
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT file_name, predicted_class, confidence, created_at
            FROM scan_records
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows
