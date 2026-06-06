import hashlib
import hmac
import os

import mysql.connector

from db import get_connection


def _hash_password(password, salt=None):
    salt = salt or os.urandom(16).hex()
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"{salt}${password_hash}"


def _verify_password(password, stored_hash):
    try:
        salt, expected_hash = stored_hash.split("$", 1)
    except ValueError:
        return False

    candidate_hash = _hash_password(password, salt).split("$", 1)[1]
    return hmac.compare_digest(candidate_hash, expected_hash)


def create_user(name, email, password):
    email = email.strip().lower()
    name = name.strip()

    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (%s, %s, %s)
                """,
                (name, email, _hash_password(password)),
            )
            conn.commit()
            user_id = cursor.lastrowid
            return {"id": user_id, "name": name, "email": email}
        except mysql.connector.IntegrityError:
            return None
        finally:
            cursor.close()


def authenticate_user(email, password):
    email = email.strip().lower()

    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, name, email, password_hash
            FROM users
            WHERE email = %s
            """,
            (email,),
        )
        user = cursor.fetchone()
        cursor.close()

    if not user or not _verify_password(password, user["password_hash"]):
        return None

    return {"id": user["id"], "name": user["name"], "email": user["email"]}


def update_user_name(user_id, name):
    name = name.strip()
    if not name:
        return False

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET name = %s
            WHERE id = %s
            """,
            (name, user_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        return updated


def update_user_password(user_id, current_password, new_password):
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT password_hash
            FROM users
            WHERE id = %s
            """,
            (user_id,),
        )
        user = cursor.fetchone()

        if not user or not _verify_password(current_password, user["password_hash"]):
            cursor.close()
            return False

        cursor.execute(
            """
            UPDATE users
            SET password_hash = %s
            WHERE id = %s
            """,
            (_hash_password(new_password), user_id),
        )
        conn.commit()
        cursor.close()
        return True


def delete_user(user_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
