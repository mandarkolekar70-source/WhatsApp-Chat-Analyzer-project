"""
AI-Powered RAG Engine for WhatsApp Chat Analysis

Implements Retrieval-Augmented Generation (RAG) using:
    - HuggingFace Embeddings (all-MiniLM-L6-v2)
    - FAISS vector store (persisted per group)
    - LangChain orchestration
    - Google Gemini 1.5 Flash for generation

Architecture:
    1. Embed messages → FAISS index
    2. Persist index to disk (one per ChatGroup)
    3. Query: Retrieve relevant messages → Pass to Gemini → Generate answer
"""

import os
from typing import List, Optional, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging

logger = logging.getLogger(__name__)


class ChatRAGEngine:
    """
    RAG engine for conversational AI over WhatsApp chat data.
    
    Each ChatGroup gets its own FAISS index stored on disk.
    Supports natural language queries about chat history.
    """
    
    FAISS_INDEX_DIR = "data/faiss_indexes"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, group_id: int, api_key: str):
        """
        Initialize RAG engine for a specific group.
        
        Args:
            group_id: Database ID of ChatGroup
            api_key: Groq API key
        """
        self.group_id = group_id
        self.api_key = api_key
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Groq LLM - Super fast and reliable!
        from langchain_openai import ChatOpenAI
        
        self.llm = ChatOpenAI(
            model="llama-3.3-70b-versatile",  # Fast and capable model
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=1024  # Increased for more detailed responses
        )
        
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain = None
        
        # Ensure index directory exists
        os.makedirs(self.FAISS_INDEX_DIR, exist_ok=True)
    
    def _get_index_path(self) -> str:
        """Get file path for this group's FAISS index."""
        return os.path.join(self.FAISS_INDEX_DIR, f"group_{self.group_id}")
    
    def build_index(self, messages: List[dict]):
        """
        Build FAISS index from chat messages.
        
        Args:
            messages: List of dicts with keys [timestamp, sender, message_text]
        """
        if not messages:
            logger.warning(f"No messages to index for group {self.group_id}")
            return
        
        # Convert messages to LangChain Documents
        documents = []
        for msg in messages:
            # Create rich document with metadata
            content = f"{msg['sender']}: {msg['message_text']}"
            metadata = {
                'sender': msg['sender'],
                'timestamp': str(msg['timestamp']),
                'date': msg['timestamp'].strftime('%Y-%m-%d'),
                'time': msg['timestamp'].strftime('%H:%M')
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Persist to disk
        index_path = self._get_index_path()
        self.vectorstore.save_local(index_path)
        
        logger.info(f"Built and saved FAISS index with {len(documents)} documents at {index_path}")
    
    def load_index(self) -> bool:
        """
        Load existing FAISS index from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        index_path = self._get_index_path()
        
        if not os.path.exists(index_path):
            logger.warning(f"No FAISS index found at {index_path}")
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for FAISS
            )
            logger.info(f"Loaded FAISS index from {index_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def initialize_qa_chain(self):
        """
        Initialize the QA chain using modern LCEL approach.
        
        Uses:
            - Vector retrieval from FAISS
            - LCEL chain composition
            - Gemini for generation
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call build_index() or load_index() first.")
        
        # Configure retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Retrieve top 10 relevant messages for comprehensive context
        )
        
        # Create prompt template with better formatting instructions
        template = """You are an intelligent WhatsApp chat analyzer. Your task is to provide comprehensive, well-structured answers based on the chat history.

**INSTRUCTIONS:**
1. Analyze ALL the provided chat messages carefully
2. Provide detailed, informative answers with proper structure
3. Use markdown formatting for better readability:
   - Use **bold** for emphasis
   - Use bullet points (•) or numbered lists for multiple items
   - Use line breaks to separate sections
   - Include relevant quotes from the chat when helpful
4. Always mention WHO said WHAT when relevant
5. Include timestamps or time context when it adds value
6. If asking about conversations/topics, summarize the key points from ALL relevant messages
7. Be thorough - don't just give brief answers, provide context and details
8. If you cannot find the answer in the provided messages, clearly state: "I don't have enough information in this chat to answer that question."

**CHAT MESSAGES:**
{context}

**QUESTION:** {question}

**ANSWER:**"""
        
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Build the chain using LCEL
        def format_docs(docs):
            formatted = []
            for doc in docs:
                sender = doc.metadata.get('sender', 'Unknown')
                timestamp = doc.metadata.get('timestamp', '')
                date_info = doc.metadata.get('date', '')
                time_info = doc.metadata.get('time', '')
                
                # Include rich context with timestamps
                msg = f"[{date_info} {time_info}] {sender}: {doc.page_content}"
                formatted.append(msg)
            
            return "\n\n".join(formatted)
        
        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Initialized QA chain using LCEL")
    
    def query(self, question: str, chat_history: List[Dict[str, str]] = None) -> dict:
        """
        Ask a question about the chat data with conversation memory.
        
        Args:
            question: Natural language question
            chat_history: Optional list of previous Q&A pairs [{'role': 'user'/'assistant', 'content': '...'}]
        
        Returns:
            dict: {
                'answer': Generated answer,
                'sources': List of source messages used
            }
        """
        if not self.qa_chain:
            self.initialize_qa_chain()
        
        # Build enhanced question with conversation history context
        enhanced_question = question
        if chat_history and len(chat_history) > 0:
            # Include last few exchanges for context (limit to last 6 messages = 3 Q&A pairs)
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            
            history_context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_history
            ])
            
            enhanced_question = f"""Previous conversation:
{history_context}

Current question: {question}"""
        
        # Get response from chain
        answer = self.qa_chain.invoke(enhanced_question)
        
        # Get source documents using invoke
        source_docs = self.retriever.invoke(question)
        
        # Extract sources - show top 5 now since we retrieve more
        sources = []
        for doc in source_docs[:5]:
            sources.append({
                'content': doc.page_content,
                'sender': doc.metadata.get('sender', 'Unknown'),
                'timestamp': doc.metadata.get('timestamp', ''),
                'date': doc.metadata.get('date', ''),
                'time': doc.metadata.get('time', '')
            })
        
        return {
            'answer': answer,
            'sources': sources
        }


def create_or_load_rag_engine(group_id: int, session, groq_api_key: str, force_rebuild: bool = False):
    """
    Factory function to create or load RAG engine for a group.
    
    Args:
        group_id: ChatGroup ID
        session: SQLAlchemy session
        groq_api_key: Groq API key
        force_rebuild: If True, rebuild index even if it exists
    
    Returns:
        ChatRAGEngine: Initialized RAG engine
    """
    from src.models import Message
    
    engine = ChatRAGEngine(group_id, groq_api_key)
    
    # Try to load existing index
    if not force_rebuild and engine.load_index():
        engine.initialize_qa_chain()
        return engine
    
    # Build new index
    messages = session.query(Message).filter_by(group_id=group_id).order_by(Message.timestamp).all()
    
    message_dicts = [
        {
            'timestamp': msg.timestamp,
            'sender': msg.sender,
            'message_text': msg.message_text
        }
        for msg in messages
    ]
    
    engine.build_index(message_dicts)
    engine.initialize_qa_chain()
    
    return engine