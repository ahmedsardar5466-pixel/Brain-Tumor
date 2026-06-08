import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

import mysql.connector

from db import get_connection

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'-]*[A-Za-z]$")


def is_valid_email(email):
    email = email.strip()
    if not email or len(email) > 150:
        return False
    if ".." in email:
        return False
    return bool(EMAIL_PATTERN.fullmatch(email))


def is_valid_name(name):
    name = " ".join(name.strip().split())
    if len(name) < 3 or len(name) > 80:
        return False
    return bool(NAME_PATTERN.fullmatch(name))


def get_password_validation_error(password):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must include at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must include at least one number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must include at least one special character."
    return None


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


def _hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_login_session(user_id):
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_sessions (user_id, token_hash, expires_at)
            VALUES (%s, %s, %s)
            """,
            (user_id, token_hash, expires_at.replace(tzinfo=None)),
        )
        conn.commit()
        cursor.close()

    return token


def get_user_by_session_token(token):
    if not token:
        return None

    token_hash = _hash_token(token)

    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("DELETE FROM user_sessions WHERE expires_at <= UTC_TIMESTAMP()")
        cursor.execute(
            """
            SELECT u.id, u.name, u.email
            FROM user_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = %s
              AND s.expires_at > UTC_TIMESTAMP()
            """,
            (token_hash,),
        )
        user = cursor.fetchone()
        conn.commit()
        cursor.close()

    return user


def delete_login_session(token):
    if not token:
        return

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE token_hash = %s", (_hash_token(token),))
        conn.commit()
        cursor.close()


def create_user(name, email, password):
    email = email.strip().lower()
    name = name.strip()

    if not is_valid_email(email):
        return None
    if not is_valid_name(name):
        return None
    if get_password_validation_error(password):
        return None

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
    name = " ".join(name.strip().split())
    if not is_valid_name(name):
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
    if get_password_validation_error(new_password):
        return False

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
