import hashlib
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

# Define the database file name
DB_FILE = "app_data.db"


class AuthLogDB:
    """
    Manages SQLite database operations for user authentication and application logging.
    - Users are stored with salted SHA-256 hashed passwords for security.
    - Logs are stored with a timestamp, event details, and the associated user ID.
    """

    def __init__(self, db_file: str = DB_FILE):
        """Initializes the database connection and ensures tables exist."""
        self.db_file = db_file
        # Establish connection. check_same_thread=False is often needed for multi-threaded/async apps,
        # but for a simple script, it's safer to keep it True or use a connection pool/context manager.
        # We will keep it simple and handle closing connections explicitly or in a context manager.
        self.conn = self._get_connection()
        self.create_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a new database connection."""
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row  # Allows accessing columns by name
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def create_tables(self) -> None:
        """Creates the 'users' and 'logs' tables if they don't exist."""
        try:
            cursor = self.conn.cursor()

            # 1. Users Table for Credentials
            # Note: We store the 'salt' used for hashing the password.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL
                )
            """
            )

            # 2. Logs Table for application activity
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id INTEGER,
                    event TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """
            )

            self.conn.commit()
            print(
                f"Database '{self.db_file}' and tables (users, logs) initialized successfully."
            )

        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
            self.conn.rollback()

    # --- Utility Functions for Hashing and Salting ---

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        """Hashes the password using the provided salt and SHA-256."""
        # Encode password to bytes and combine with salt
        password_bytes = password.encode("utf-8")
        hashed_bytes = hashlib.sha256(salt + password_bytes).hexdigest()
        return hashed_bytes

    @staticmethod
    def _generate_salt() -> str:
        """Generates a random salt (16 bytes) and returns it as a hex string."""
        return os.urandom(16).hex()

    # --- User Management Functions ---

    def register_user(self, username: str, password: str) -> Optional[int]:
        """
        Registers a new user, hashing the password with a unique salt.
        Returns the new user's ID or None on failure.
        """
        if not username or not password:
            print("Username and password cannot be empty.")
            return None

        salt_hex = self._generate_salt()
        salt_bytes = bytes.fromhex(salt_hex)
        password_hash = self._hash_password(password, salt_bytes)

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, password_hash, salt_hex),
            )
            self.conn.commit()
            user_id = cursor.lastrowid
            self.log_event(user_id, f"User '{username}' registered successfully.")
            return user_id
        except sqlite3.IntegrityError:
            print(f"Registration failed: Username '{username}' already exists.")
            return None
        except sqlite3.Error as e:
            print(f"Database error during registration: {e}")
            self.conn.rollback()
            return None

    def verify_user(self, username: str, password: str) -> Optional[int]:
        """
        Verifies user credentials.
        Returns the user's ID on success, None otherwise.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, password_hash, salt FROM users WHERE username = ?",
                (username,),
            )
            user_record = cursor.fetchone()

            if user_record:
                # Retrieve stored salt (convert hex string back to bytes)
                stored_salt = bytes.fromhex(user_record["salt"])

                # Hash the provided password with the stored salt
                input_hash = self._hash_password(password, stored_salt)

                # Compare the generated hash with the stored hash
                if input_hash == user_record["password_hash"]:
                    user_id = user_record["id"]
                    self.log_event(
                        user_id, f"User '{username}' logged in successfully."
                    )
                    return user_id

            # If user_record is None or password hash doesn't match
            print(f"Verification failed for user '{username}'.")
            return None

        except sqlite3.Error as e:
            print(f"Database error during verification: {e}")
            return None

    # --- Logging Functions ---

    def log_event(self, user_id: Optional[int], event: str) -> None:
        """
        Records an event in the logs table.
        user_id can be None for system or unauthenticated events.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO logs (timestamp, user_id, event) VALUES (?, ?, ?)",
                (timestamp, user_id, event),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging event: {e}")
            self.conn.rollback()

    def get_logs(self) -> List[Dict[str, Any]]:
        """Retrieves all log entries, joining with the users table to show usernames."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT
                    l.timestamp,
                    COALESCE(u.username, 'SYSTEM/UNAUTH') as username,
                    l.event
                FROM logs l
                LEFT JOIN users u ON l.user_id = u.id
                ORDER BY l.timestamp DESC
            """
            )

            # Fetch all results as a list of dictionaries (due to row_factory=sqlite3.Row)
            logs = [dict(row) for row in cursor.fetchall()]
            return logs

        except sqlite3.Error as e:
            print(f"Error retrieving logs: {e}")
            return []

    def close(self) -> None:
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print(f"Connection to '{self.db_file}' closed.")


# --- Example Usage ---

if __name__ == "__main__":

    # 1. Clean up old database file for a fresh start (optional)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"\n--- Removed old '{DB_FILE}' for demonstration. ---\n")

    # 2. Initialize the Database Manager
    db_manager = AuthLogDB()

    # Log an unauthenticated event
    db_manager.log_event(None, "Application started.")

    # 3. User Registration
    print("\n--- Registering Users ---")
    user_id_john = db_manager.register_user("john_doe", "SecureP@ss123")
    user_id_jane = db_manager.register_user("jane_smith", "MyStrongPwd456")
    db_manager.register_user(
        "john_doe", "another_attempt"
    )  # Should fail (IntegrityError)

    # 4. User Verification (Login)
    print("\n--- Verifying Users ---")

    # Successful login
    logged_in_user_id = db_manager.verify_user("john_doe", "SecureP@ss123")
    if logged_in_user_id:
        print(f"Login SUCCESS for John Doe. User ID: {logged_in_user_id}")
        db_manager.log_event(logged_in_user_id, "Accessed admin panel.")
    else:
        print("Login FAILED for John Doe.")

    # Failed login
    failed_id = db_manager.verify_user("john_doe", "WrongPassword")
    if failed_id is None:
        print("Login FAILED as expected for wrong password.")

    # Non-existent user
    db_manager.verify_user("non_existent", "password")

    # 5. Retrieve Logs
    print("\n--- Application Logs ---")
    all_logs = db_manager.get_logs()

    if all_logs:
        for log in all_logs:
            # Print logs in a readable format
            print(
                f"[{log['timestamp']}] | User: {log['username']} | Event: {log['event']}"
            )
    else:
        print("No logs found.")

    # 6. Close Connection
    db_manager.close()
