"""
Database Connection & Session Management

This module provides a singleton-style database engine and session factory.
It ensures proper connection pooling and session lifecycle management.

Best Practices:
    - Use context managers for session handling
    - Never commit sessions in service layer (let caller decide)
    - Engine is created once and reused across the application
"""

import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from src.models import Base
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Singleton database manager.
    
    Manages SQLAlchemy engine creation, connection pooling,
    and session factory for the entire application.
    """
    
    _engine = None
    _SessionFactory = None
    
    @classmethod
    def get_engine(cls):
        """
        Create or retrieve the database engine.
        
        Uses connection pooling for production-grade performance.
        Connection string should be provided via environment variable.
        
        Returns:
            sqlalchemy.engine.Engine: Database engine instance
        """
        if cls._engine is None:
            # Default to localhost PostgreSQL
            # In production, use: os.getenv('DATABASE_URL')
            DATABASE_URL = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:postgres@localhost:5432/socialpulse'
            )
            
            cls._engine = create_engine(
                DATABASE_URL,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                echo=False  # Set to True for SQL query logging in development
            )
            
            logger.info(f"Database engine created: {DATABASE_URL.split('@')[1]}")
        
        return cls._engine
    
    @classmethod
    def get_session_factory(cls):
        """
        Get or create the session factory.
        
        Returns:
            sessionmaker: Factory for creating database sessions
        """
        if cls._SessionFactory is None:
            engine = cls.get_engine()
            cls._SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
        
        return cls._SessionFactory
    
    @classmethod
    def init_db(cls):
        """
        Initialize database schema.
        
        Creates all tables defined in models.py if they don't exist.
        Safe to call multiple times (idempotent).
        """
        engine = cls.get_engine()
        Base.metadata.create_all(engine)
        logger.info("Database schema initialized successfully")
    
    @classmethod
    @contextmanager
    def get_session(cls) -> Session:
        """
        Context manager for database sessions.
        
        Ensures proper session lifecycle:
            - Automatic commit on success
            - Automatic rollback on exception
            - Always closes session
        
        Usage:
            with DatabaseManager.get_session() as session:
                user = session.query(User).first()
        
        Yields:
            Session: SQLAlchemy session
        """
        SessionFactory = cls.get_session_factory()
        session = SessionFactory()
        
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session rollback due to error: {e}")
            raise
        finally:
            session.close()


# Convenience functions for cleaner imports
def get_db_session():
    """Get a new database session context manager."""
    return DatabaseManager.get_session()


def init_database():
    """Initialize the database schema."""
    DatabaseManager.init_db()


# Initialize database on module import (safe for production)
# Alternatively, call this explicitly in app.py startup
if __name__ == "__main__":
    init_database()
    print("✅ Database initialized successfully")