"""
Analytics Engine for SocialPulse

Implements SQL-based analytics and NetworkX graph analysis.

Key Features:
    1. SQL-first approach for aggregations (no pandas groupby)
    2. NetworkX directed graph for social influence analysis
    3. PageRank-based influence scoring
    
Graph Theory Heuristic (Interview-Critical):
    If User B sends a message within 120 seconds AFTER User A,
    we create a directed edge A → B, representing that B is
    "responding to" or "influenced by" A's message.
    
    This 120-second window is a domain-specific threshold that
    balances real conversation flow vs. coincidental timing.
"""

import networkx as nx
from sqlalchemy import text
from typing import Dict, List, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SocialGraphAnalyzer:
    """
    Builds and analyzes social interaction graphs from chat data.
    
    Uses NetworkX to model users as nodes and replies as directed edges.
    Applies PageRank algorithm to identify influential users.
    """
    
    # Heuristic constant: maximum time gap for considering a reply
    REPLY_WINDOW_SECONDS = 120
    
    def __init__(self, group_id: int, session):
        """
        Initialize analyzer for a specific chat group.
        
        Args:
            group_id: Database ID of the ChatGroup
            session: SQLAlchemy session
        """
        self.group_id = group_id
        self.session = session
        self.graph = nx.DiGraph()
    
    def build_interaction_graph(self) -> nx.DiGraph:
        """
        Build directed graph of user interactions.
        
        Algorithm:
            1. Load messages ordered by timestamp
            2. For each message, look at the previous message
            3. If time difference ≤ 120 seconds AND different sender:
               Create edge: previous_sender → current_sender
            4. Edge weight = number of such interactions
        
        Why 120 seconds?
            - Typical conversation response time in group chats
            - Filters out coincidental timing
            - Captures intentional engagement
        
        Returns:
            nx.DiGraph: Directed graph with weighted edges
        """
        from src.models import Message
        
        # Load messages ordered by timestamp
        messages = self.session.query(Message).filter_by(
            group_id=self.group_id
        ).order_by(Message.timestamp).all()
        
        if not messages:
            logger.warning(f"No messages found for group {self.group_id}")
            return self.graph
        
        # Add all users as nodes
        unique_senders = set(msg.sender for msg in messages)
        self.graph.add_nodes_from(unique_senders)
        
        # Build edges based on reply heuristic
        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]
            
            # Calculate time difference in seconds
            time_diff = (curr_msg.timestamp - prev_msg.timestamp).total_seconds()
            
            # Apply 120-second reply heuristic
            if (time_diff <= self.REPLY_WINDOW_SECONDS and 
                prev_msg.sender != curr_msg.sender):
                
                # Create/strengthen directed edge
                if self.graph.has_edge(prev_msg.sender, curr_msg.sender):
                    self.graph[prev_msg.sender][curr_msg.sender]['weight'] += 1
                else:
                    self.graph.add_edge(prev_msg.sender, curr_msg.sender, weight=1)
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def calculate_influence_scores(self) -> Dict[str, float]:
        """
        Calculate PageRank scores for all users.
        
        PageRank interprets the graph as:
            - High score = Many users respond to this person
            - Low score = This person mostly responds to others
        
        Returns:
            Dict[str, float]: Username → PageRank score mapping
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Use weighted PageRank
        pagerank_scores = nx.pagerank(self.graph, weight='weight')
        
        # Sort by score descending
        sorted_scores = dict(sorted(
            pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_scores
    
    def get_top_influencers(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N most influential users.
        
        Args:
            top_n: Number of top users to return
        
        Returns:
            List of (username, score) tuples
        """
        scores = self.calculate_influence_scores()
        return list(scores.items())[:top_n]


class SQLAnalytics:
    """
    SQL-based analytics queries.
    
    Demonstrates database-level analytical skills instead of
    relying on pandas groupby operations.
    """
    
    @staticmethod
    def get_top_active_users(session, group_id: int, top_n: int = 5) -> pd.DataFrame:
        """
        Get top N most active users in a group using raw SQL.
        
        Query Explanation:
            - COUNT messages per sender
            - ORDER BY count descending
            - LIMIT to top N
        
        Args:
            session: SQLAlchemy session
            group_id: ChatGroup ID
            top_n: Number of users to return
        
        Returns:
            pd.DataFrame: Columns [sender, message_count]
        """
        query = text("""
            SELECT sender, COUNT(*) as message_count
            FROM messages
            WHERE group_id = :group_id
            GROUP BY sender
            ORDER BY message_count DESC
            LIMIT :top_n
        """)
        
        result = session.execute(query, {'group_id': group_id, 'top_n': top_n})
        
        df = pd.DataFrame(result.fetchall(), columns=['sender', 'message_count'])
        return df
    
    @staticmethod
    def get_hourly_activity(session, group_id: int) -> pd.DataFrame:
        """
        Get message count distribution by hour of day.
        
        SQL Technique:
            - EXTRACT hour from timestamp
            - GROUP BY hour
            - Return hourly counts for heatmap visualization
        
        Args:
            session: SQLAlchemy session
            group_id: ChatGroup ID
        
        Returns:
            pd.DataFrame: Columns [hour, message_count]
        """
        query = text("""
            SELECT EXTRACT(HOUR FROM timestamp) as hour, COUNT(*) as message_count
            FROM messages
            WHERE group_id = :group_id
            GROUP BY hour
            ORDER BY hour
        """)
        
        result = session.execute(query, {'group_id': group_id})
        
        df = pd.DataFrame(result.fetchall(), columns=['hour', 'message_count'])
        return df
    
    @staticmethod
    def get_daily_activity(session, group_id: int) -> pd.DataFrame:
        """
        Get message count by date.
        
        Args:
            session: SQLAlchemy session
            group_id: ChatGroup ID
        
        Returns:
            pd.DataFrame: Columns [date, message_count]
        """
        query = text("""
            SELECT DATE(timestamp) as date, COUNT(*) as message_count
            FROM messages
            WHERE group_id = :group_id
            GROUP BY date
            ORDER BY date
        """)
        
        result = session.execute(query, {'group_id': group_id})
        
        df = pd.DataFrame(result.fetchall(), columns=['date', 'message_count'])
        return df
    
    @staticmethod
    def get_total_message_count(session, group_id: int) -> int:
        """Get total message count for a group."""
        query = text("""
            SELECT COUNT(*) as total
            FROM messages
            WHERE group_id = :group_id
        """)
        
        result = session.execute(query, {'group_id': group_id})
        return result.fetchone()[0]
    
    @staticmethod
    def get_unique_sender_count(session, group_id: int) -> int:
        """Get count of unique participants."""
        query = text("""
            SELECT COUNT(DISTINCT sender) as unique_senders
            FROM messages
            WHERE group_id = :group_id
        """)
        
        result = session.execute(query, {'group_id': group_id})
        return result.fetchone()[0]
    
    @staticmethod
    def get_emoji_statistics(session, group_id: int) -> pd.DataFrame:
        """Get emoji usage statistics."""
        from src.models import Message
        import emoji
        import re
        
        messages = session.query(Message).filter_by(group_id=group_id).all()
        
        emoji_counts = {}
        for msg in messages:
            if msg.message_text:
                emojis = emoji.emoji_list(msg.message_text)
                for em in emojis:
                    emoji_char = em['emoji']
                    emoji_counts[emoji_char] = emoji_counts.get(emoji_char, 0) + 1
        
        df = pd.DataFrame(list(emoji_counts.items()), columns=['emoji', 'count'])
        df = df.sort_values('count', ascending=False).head(20)
        return df
    
    @staticmethod
    def get_word_frequency(session, group_id: int, top_n: int = 50) -> pd.DataFrame:
        """Get most common words (excluding stop words)."""
        from src.models import Message
        import re
        from collections import Counter
        
        messages = session.query(Message).filter_by(group_id=group_id).all()
        
        # Common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 
                     'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'are', 'was', 'were'}
        
        words = []
        for msg in messages:
            if msg.message_text:
                # Extract words, convert to lowercase
                text_words = re.findall(r'\b[a-zA-Z]{3,}\b', msg.message_text.lower())
                words.extend([w for w in text_words if w not in stop_words])
        
        word_counts = Counter(words).most_common(top_n)
        df = pd.DataFrame(word_counts, columns=['word', 'count'])
        return df
    
    @staticmethod
    def get_message_length_stats(session, group_id: int) -> pd.DataFrame:
        """Get statistics about message lengths by user."""
        query = text("""
            SELECT sender, 
                   AVG(LENGTH(message_text)) as avg_length,
                   MAX(LENGTH(message_text)) as max_length,
                   COUNT(*) as message_count
            FROM messages
            WHERE group_id = :group_id AND message_text IS NOT NULL
            GROUP BY sender
            ORDER BY avg_length DESC
        """)
        
        result = session.execute(query, {'group_id': group_id})
        df = pd.DataFrame(result.fetchall(), 
                         columns=['sender', 'avg_length', 'max_length', 'message_count'])
        return df
    
    @staticmethod
    def get_weekly_activity(session, group_id: int) -> pd.DataFrame:
        """Get message count by day of week."""
        query = text("""
            SELECT 
                CASE EXTRACT(DOW FROM timestamp)
                    WHEN 0 THEN 'Sunday'
                    WHEN 1 THEN 'Monday'
                    WHEN 2 THEN 'Tuesday'
                    WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday'
                    WHEN 5 THEN 'Friday'
                    WHEN 6 THEN 'Saturday'
                END as day_of_week,
                EXTRACT(DOW FROM timestamp) as day_num,
                COUNT(*) as message_count
            FROM messages
            WHERE group_id = :group_id
            GROUP BY day_of_week, day_num
            ORDER BY day_num
        """)
        
        result = session.execute(query, {'group_id': group_id})
        df = pd.DataFrame(result.fetchall(), columns=['day_of_week', 'day_num', 'message_count'])
        return df
    
    @staticmethod
    def get_response_time_analysis(session, group_id: int) -> pd.DataFrame:
        """Analyze average response times between users."""
        from src.models import Message
        
        messages = session.query(Message).filter_by(
            group_id=group_id
        ).order_by(Message.timestamp).all()
        
        response_times = []
        
        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]
            
            if prev_msg.sender != curr_msg.sender:
                time_diff = (curr_msg.timestamp - prev_msg.timestamp).total_seconds() / 60  # in minutes
                if time_diff <= 60:  # Only consider responses within 1 hour
                    response_times.append({
                        'from_user': prev_msg.sender,
                        'to_user': curr_msg.sender,
                        'response_time_minutes': time_diff
                    })
        
        if not response_times:
            return pd.DataFrame(columns=['from_user', 'to_user', 'avg_response_time'])
        
        df = pd.DataFrame(response_times)
        avg_response = df.groupby(['from_user', 'to_user'])['response_time_minutes'].mean().reset_index()
        avg_response.columns = ['from_user', 'to_user', 'avg_response_time']
        avg_response = avg_response.sort_values('avg_response_time').head(10)
        
        return avg_response