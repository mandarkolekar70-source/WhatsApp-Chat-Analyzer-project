"""
SQLAlchemy ORM Models for SocialPulse SaaS

This module defines the database schema for a multi-user WhatsApp analytics platform.
The design follows normalized relational database principles with proper foreign key relationships.

Tables:
    - User: Authentication and user management
    - ChatGroup: WhatsApp group metadata linked to users
    - Message: Individual chat messages with sentiment analysis
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text, Index
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    """
    User authentication table.
    
    Attributes:
        id: Primary key
        username: Unique username (indexed for fast lookup during login)
        password_hash: bcrypt-hashed password (never store plaintext)
        created_at: Account creation timestamp
    
    Relationships:
        chat_groups: One-to-many relationship with ChatGroup
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    chat_groups = relationship('ChatGroup', back_populates='user', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class ChatGroup(Base):
    """
    WhatsApp group metadata table.
    
    Each uploaded chat file becomes a ChatGroup entry.
    Multiple groups can belong to the same user.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User (indexed for filtering queries)
        group_name: Display name of the WhatsApp group
        uploaded_at: Timestamp when the chat was uploaded
    
    Relationships:
        user: Many-to-one relationship with User
        messages: One-to-many relationship with Message
    """
    __tablename__ = 'chat_groups'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    group_name = Column(String(100), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship('User', back_populates='chat_groups')
    messages = relationship('Message', back_populates='chat_group', cascade='all, delete-orphan')
    
    # Composite index for efficient user-specific group queries
    __table_args__ = (
        Index('idx_user_uploaded', 'user_id', 'uploaded_at'),
    )
    
    def __repr__(self):
        return f"<ChatGroup(id={self.id}, name='{self.group_name}', user_id={self.user_id})>"


class Message(Base):
    """
    Individual WhatsApp messages.
    
    Stores parsed chat messages with metadata and sentiment analysis.
    Large table optimized with indexes on frequently queried columns.
    
    Attributes:
        id: Primary key
        group_id: Foreign key to ChatGroup (indexed for group-level queries)
        timestamp: When the message was sent
        sender: Name of the person who sent the message (indexed for analytics)
        message_text: Full text content
        sentiment_score: Pre-computed sentiment (-1.0 to 1.0, nullable for future enhancement)
    
    Relationships:
        chat_group: Many-to-one relationship with ChatGroup
    """
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(Integer, ForeignKey('chat_groups.id', ondelete='CASCADE'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)  # Indexed for time-based queries
    sender = Column(String(100), nullable=False, index=True)  # Indexed for user analytics
    message_text = Column(Text, nullable=False)
    sentiment_score = Column(Float, nullable=True)  # Nullable for phased implementation
    
    # Relationships
    chat_group = relationship('ChatGroup', back_populates='messages')
    
    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_group_timestamp', 'group_id', 'timestamp'),
        Index('idx_group_sender', 'group_id', 'sender'),
    )
    
    def __repr__(self):
        return f"<Message(id={self.id}, sender='{self.sender}', timestamp={self.timestamp})>"