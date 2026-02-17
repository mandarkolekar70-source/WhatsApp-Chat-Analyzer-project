"""
WhatsApp Chat Parser

Robust parser for WhatsApp exported chat files.
Handles multiple date formats, multi-line messages, emojis, and special characters.

Supported Format:
    dd/mm/yy, hh:mm - Name: Message

Design Decisions:
    - Uses regex for reliable pattern matching
    - Filters out system messages (media notifications, group events)
    - Handles multi-line messages by accumulating text
    - Returns clean pandas DataFrame ready for database insertion
"""

import re
import pandas as pd
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class WhatsAppParser:
    """
    Parser for WhatsApp chat export files.
    
    Attributes:
        MESSAGE_PATTERN: Regex pattern for message lines
        SYSTEM_KEYWORDS: Patterns indicating system/media messages to filter
    """
    
    # Regex pattern for WhatsApp message format: dd/mm/yy, hh:mm - Name: Message
    # Captures: date, time, sender, message
    MESSAGE_PATTERN = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]m)?)\s-\s([^:]+):\s(.+)',
        re.IGNORECASE
    )
    
    # System message indicators to filter out
    SYSTEM_KEYWORDS = [
        '<Media omitted>',
        'image omitted',
        'video omitted',
        'audio omitted',
        'document omitted',
        'sticker omitted',
        'GIF omitted',
        'Messages and calls are end-to-end encrypted',
        'created group',
        'added',
        'removed',
        'left',
        'changed the subject',
        'changed this group\'s icon',
        'deleted this message',
        'This message was deleted'
    ]
    
    def __init__(self):
        """Initialize the parser."""
        self.parsed_messages = []
    
    def is_system_message(self, message_text: str) -> bool:
        """
        Check if a message is a system notification.
        
        Args:
            message_text: The message content
        
        Returns:
            bool: True if system message, False otherwise
        """
        message_lower = message_text.lower()
        return any(keyword.lower() in message_lower for keyword in self.SYSTEM_KEYWORDS)
    
    def parse_datetime(self, date_str: str, time_str: str) -> datetime:
        """
        Parse WhatsApp date and time strings into datetime object.
        
        Handles multiple formats:
            - dd/mm/yy or dd/mm/yyyy
            - 24-hour or 12-hour time formats
        
        Args:
            date_str: Date string (e.g., "25/12/23")
            time_str: Time string (e.g., "14:30" or "2:30 pm")
        
        Returns:
            datetime: Parsed datetime object
        """
        # Normalize time string
        time_str = time_str.strip().lower()
        
        # Try different datetime formats
        formats = [
            "%d/%m/%y, %H:%M:%S",
            "%d/%m/%y, %H:%M",
            "%d/%m/%Y, %H:%M:%S",
            "%d/%m/%Y, %H:%M",
            "%d/%m/%y, %I:%M:%S %p",
            "%d/%m/%y, %I:%M %p",
            "%d/%m/%Y, %I:%M:%S %p",
            "%d/%m/%Y, %I:%M %p",
        ]
        
        combined = f"{date_str}, {time_str}"
        
        for fmt in formats:
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue
        
        # Fallback: try without seconds
        raise ValueError(f"Unable to parse datetime: {combined}")
    
    def parse_file(self, file_content: str) -> pd.DataFrame:
        """
        Parse WhatsApp chat file content into structured DataFrame.
        
        Algorithm:
            1. Split file into lines
            2. Identify message start lines using regex
            3. Accumulate multi-line messages
            4. Filter system messages
            5. Return clean DataFrame
        
        Args:
            file_content: Full text content of WhatsApp export
        
        Returns:
            pd.DataFrame: Columns [timestamp, sender, message_text]
        
        Raises:
            ValueError: If no valid messages found
        """
        lines = file_content.split('\n')
        messages = []
        current_message = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a new message
            match = self.MESSAGE_PATTERN.match(line)
            
            if match:
                # Save previous message if exists
                if current_message:
                    messages.append(current_message)
                
                # Start new message
                date_str, time_str, sender, message_text = match.groups()
                
                try:
                    timestamp = self.parse_datetime(date_str, time_str)
                    
                    current_message = {
                        'timestamp': timestamp,
                        'sender': sender.strip(),
                        'message_text': message_text.strip()
                    }
                except ValueError as e:
                    logger.warning(f"Skipping message with invalid datetime: {e}")
                    current_message = None
            
            else:
                # Continuation of previous message (multi-line)
                if current_message:
                    current_message['message_text'] += f"\n{line}"
        
        # Add last message
        if current_message:
            messages.append(current_message)
        
        if not messages:
            raise ValueError("No valid messages found in the file")
        
        # Create DataFrame
        df = pd.DataFrame(messages)
        
        # Filter out system messages
        df = df[~df['message_text'].apply(self.is_system_message)]
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Parsed {len(df)} valid messages from chat export")
        
        return df


def parse_whatsapp_file(file_content: str) -> pd.DataFrame:
    """
    Convenience function to parse WhatsApp file.
    
    Args:
        file_content: Text content of WhatsApp export
    
    Returns:
        pd.DataFrame: Parsed messages
    """
    parser = WhatsAppParser()
    return parser.parse_file(file_content)