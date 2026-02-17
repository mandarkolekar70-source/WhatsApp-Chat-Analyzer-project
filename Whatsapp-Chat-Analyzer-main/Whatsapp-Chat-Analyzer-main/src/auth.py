"""
Authentication Utilities

Handles password hashing, verification, and user session management.
Uses bcrypt for secure password storage following industry best practices.
"""

import bcrypt
import logging

logger = logging.getLogger(__name__)


def hash_password(plain_password: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    
    Args:
        plain_password: User's plaintext password
    
    Returns:
        str: bcrypt-hashed password (suitable for database storage)
    """
    password_bytes = plain_password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a stored hash.
    
    Args:
        plain_password: Password attempt from user
        hashed_password: Stored hash from database
    
    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def create_user(session, username: str, password: str):
    """
    Create a new user with hashed password.
    
    Args:
        session: SQLAlchemy session
        username: Desired username
        password: Plaintext password
    
    Returns:
        User object if created, None if username exists
    """
    from src.models import User
    
    # Check if username already exists
    existing_user = session.query(User).filter_by(username=username).first()
    if existing_user:
        return None
    
    # Create new user
    password_hash = hash_password(password)
    new_user = User(username=username, password_hash=password_hash)
    session.add(new_user)
    session.commit()
    
    logger.info(f"User created: {username}")
    return new_user


def authenticate_user(session, username: str, password: str):
    """
    Authenticate user credentials.
    
    Args:
        session: SQLAlchemy session
        username: Username attempt
        password: Password attempt
    
    Returns:
        User object if authenticated, None otherwise
    """
    from src.models import User
    
    user = session.query(User).filter_by(username=username).first()
    
    if not user:
        return None
    
    if verify_password(password, user.password_hash):
        logger.info(f"User authenticated: {username}")
        return user
    
    return None