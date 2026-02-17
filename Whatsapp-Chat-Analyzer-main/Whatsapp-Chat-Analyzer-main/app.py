"""
SocialPulse SaaS - Multi-User WhatsApp Analytics Platform

A production-grade analytics platform with:
    - User authentication
    - WhatsApp chat parsing and storage
    - SQL-based analytics
    - Social graph analysis (NetworkX + PageRank)
    - AI-powered chat queries (RAG with Gemini)

Author: Senior Full-Stack Data Engineer
Tech Stack: Streamlit, SQLAlchemy, PostgreSQL, NetworkX, LangChain, Gemini
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import os
from dotenv import load_dotenv

# Force reload environment variables
load_dotenv(override=True)

from datetime import datetime

# Import local modules
from src.db import DatabaseManager, init_database
from src.models import User, ChatGroup, Message
from src.auth import create_user, authenticate_user
from src.parser import parse_whatsapp_file
from src.analytics import SocialGraphAnalyzer, SQLAnalytics
from src.ai_engine import create_or_load_rag_engine

# Page configuration
st.set_page_config(
    page_title="SocialPulse | WhatsApp Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database
init_database()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'selected_group_id' not in st.session_state:
    st.session_state.selected_group_id = None


def login_page():
    """
    Authentication page with Login and Signup tabs.
    """
    st.title("🔐 SocialPulse Authentication")
    
    tab1, tab2 = st.tabs(["Login", "Signup"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    with DatabaseManager.get_session() as session:
                        user = authenticate_user(session, username, password)
                        
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.user_id = user.id
                            st.session_state.username = user.username
                            st.success(f"Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_signup = st.form_submit_button("Sign Up")
            
            if submit_signup:
                if not new_username or not new_password:
                    st.error("Please fill all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    with DatabaseManager.get_session() as session:
                        user = create_user(session, new_username, new_password)
                        
                        if user:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Username already exists")


def upload_chat_section():
    """
    Chat upload and parsing interface.
    """
    st.subheader("📤 Upload WhatsApp Chat")
    
    uploaded_file = st.file_uploader(
        "Choose a WhatsApp chat export (.txt)",
        type=['txt'],
        help="Export chat from WhatsApp: Open chat → ⋮ → More → Export chat → Without media"
    )
    
    group_name = st.text_input("Group Name", placeholder="e.g., Family, Work Team, Friends")
    
    if st.button("Upload & Process"):
        if not uploaded_file:
            st.error("Please select a file")
        elif not group_name:
            st.error("Please enter a group name")
        else:
            with st.spinner("Processing chat file..."):
                try:
                    # Read file
                    file_content = uploaded_file.read().decode('utf-8')
                    
                    # Parse WhatsApp file
                    df = parse_whatsapp_file(file_content)
                    
                    st.success(f"✅ Parsed {len(df)} messages")
                    
                    # Save to database
                    with DatabaseManager.get_session() as session:
                        # Create ChatGroup entry
                        chat_group = ChatGroup(
                            user_id=st.session_state.user_id,
                            group_name=group_name
                        )
                        session.add(chat_group)
                        session.flush()  # Get ID without committing
                        
                        # Insert messages
                        for _, row in df.iterrows():
                            message = Message(
                                group_id=chat_group.id,
                                timestamp=row['timestamp'],
                                sender=row['sender'],
                                message_text=row['message_text']
                            )
                            session.add(message)
                        
                        session.commit()
                        
                        st.success(f"✅ Saved to database as '{group_name}'")
                        
                        # Build FAISS index
                        groq_api_key = os.getenv('GROQ_API_KEY')
                        if groq_api_key:
                            with st.spinner("Building AI search index..."):
                                messages = [
                                    {
                                        'timestamp': row['timestamp'],
                                        'sender': row['sender'],
                                        'message_text': row['message_text']
                                    }
                                    for _, row in df.iterrows()
                                ]
                                
                                from src.ai_engine import ChatRAGEngine
                                engine = ChatRAGEngine(chat_group.id, groq_api_key)
                                engine.build_index(messages)
                                
                                st.success("✅ AI index built successfully")
                        
                        st.balloons()
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")


def dashboard_page():
    """
    Main dashboard with sidebar and tabs.
    """
    # Sidebar
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.selected_group_id = None
            st.rerun()
        
        st.divider()
        
        # Upload section
        with st.expander("📤 Upload New Chat", expanded=False):
            upload_chat_section()
        
        st.divider()
        
        # Group selection
        st.subheader("📁 Your Chat Groups")
        
        with DatabaseManager.get_session() as session:
            groups = session.query(ChatGroup).filter_by(
                user_id=st.session_state.user_id
            ).order_by(ChatGroup.uploaded_at.desc()).all()
            
            if not groups:
                st.info("No chat groups yet. Upload one above!")
            else:
                group_options = {g.group_name: g.id for g in groups}
                selected_name = st.selectbox(
                    "Select Group",
                    options=list(group_options.keys())
                )
                
                st.session_state.selected_group_id = group_options[selected_name]
    
    # Main content
    if not st.session_state.selected_group_id:
        st.title("Welcome to SocialPulse 📊")
        st.markdown("""
        ### Get Started
        1. Upload a WhatsApp chat export using the sidebar
        2. Select a chat group to analyze
        3. Explore analytics, graphs, and AI insights
        """)
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "😊 Emoji & Words", 
        "📈 Temporal Patterns",
        "🕸️ Social Graph", 
        "🤖 AI Chat"
    ])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_emoji_words_tab()
    
    with tab3:
        show_temporal_patterns_tab()
    
    with tab4:
        show_social_graph_tab()
    
    with tab5:
        show_ai_chat_tab()


def show_overview_tab():
    """
    Overview tab with metrics and charts.
    """
    st.header("📊 Chat Overview")
    
    group_id = st.session_state.selected_group_id
    
    with DatabaseManager.get_session() as session:
        # High-level metrics
        total_messages = SQLAnalytics.get_total_message_count(session, group_id)
        unique_participants = SQLAnalytics.get_unique_sender_count(session, group_id)
        
        col1, col2 = st.columns(2)
        col1.metric("Total Messages", f"{total_messages:,}")
        col2.metric("Participants", unique_participants)
        
        st.divider()
        
        # Top active users
        st.subheader("🏆 Most Active Users")
        top_users_df = SQLAnalytics.get_top_active_users(session, group_id, top_n=5)
        
        if not top_users_df.empty:
            st.bar_chart(top_users_df.set_index('sender'))
            st.dataframe(top_users_df, use_container_width=True)
        
        st.divider()
        
        # Hourly activity heatmap
        st.subheader("⏰ Activity by Hour of Day")
        hourly_df = SQLAnalytics.get_hourly_activity(session, group_id)
        
        if not hourly_df.empty:
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(hourly_df['hour'], hourly_df['message_count'], color='#1f77b4')
            ax.set_xlabel('Hour of Day (24h)')
            ax.set_ylabel('Message Count')
            ax.set_title('Message Distribution by Hour')
            ax.set_xticks(range(24))
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)


def show_social_graph_tab():
    """
    Social graph analysis tab.
    """
    st.header("🕸️ Social Influence Graph")
    
    st.markdown("""
    **Graph Theory Analysis:**
    - Nodes represent chat participants
    - Directed edges represent influence (A → B means B responds to A)
    - Edge weight = number of reply interactions
    - **120-second heuristic:** If B messages within 120s after A, we create edge A → B
    """)
    
    group_id = st.session_state.selected_group_id
    
    with DatabaseManager.get_session() as session:
        with st.spinner("Building social graph..."):
            analyzer = SocialGraphAnalyzer(group_id, session)
            graph = analyzer.build_interaction_graph()
            
            if graph.number_of_nodes() == 0:
                st.warning("No interaction data available")
                return
            
            # PageRank scores
            st.subheader("🎯 Influence Ranking (PageRank)")
            top_influencers = analyzer.get_top_influencers(top_n=5)
            
            influence_df = pd.DataFrame(top_influencers, columns=['User', 'Influence Score'])
            influence_df['Influence Score'] = influence_df['Influence Score'].round(4)
            st.dataframe(influence_df, use_container_width=True)
            
            st.divider()
            
            # Visualize graph
            st.subheader("📈 Interactive Graph Visualization")
            
            # Prepare nodes and edges for streamlit-agraph
            nodes = []
            edges = []
            
            pagerank = nx.pagerank(graph, weight='weight')
            
            for node in graph.nodes():
                size = 20 + pagerank[node] * 500  # Scale node size by PageRank
                nodes.append(Node(
                    id=node,
                    label=node,
                    size=size,
                    color="#1f77b4"
                ))
            
            for edge in graph.edges(data=True):
                edges.append(Edge(
                    source=edge[0],
                    target=edge[1],
                    label=str(edge[2]['weight']),
                    color="#999"
                ))
            
            config = Config(
                width=800,
                height=600,
                directed=True,
                physics=True,
                hierarchical=False
            )
            
            agraph(nodes=nodes, edges=edges, config=config)


def show_ai_chat_tab():
    """
    AI-powered chat interface using RAG.
    """
    st.header("🤖 AI Chat Assistant")
    
    # Add helpful tips in an expander
    with st.expander("💡 Tips for Better Results", expanded=False):
        st.markdown("""
        **How to get the best answers:**
        - Be specific with names, dates, or topics
        - Ask one question at a time
        - The AI remembers your conversation, so you can ask follow-up questions
        - Use natural language
        
        **Example questions:**
        - "What did John say about the project deadline?"
        - "Summarize all discussions about vacation plans"
        - "Who talked about shopping and what did they say?"
        - "What happened on January 15th?"
        - "List all the links or phone numbers shared"
        - "Who are the most active participants and what do they usually talk about?"
        - "What was the conversation about yesterday?"
        - "Find all messages mentioning 'birthday' or 'celebration'"
        """)
    
    st.markdown("Ask questions about your chat data. The AI will search and analyze your messages.")
    st.caption("⚡ Optimized for fast responses | 🔒 Your data stays private | 🧠 Remembers conversation context")
    
    
    # Check for Groq API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        st.error("⚠️ Groq API key not configured. Please set GROQ_API_KEY environment variable.")
        st.code("Get your free API key at: https://console.groq.com/")
        return
    
    group_id = st.session_state.selected_group_id
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Clear chat button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Load RAG engine
    with DatabaseManager.get_session() as session:
        try:
            engine = create_or_load_rag_engine(group_id, session, groq_api_key)
        except Exception as e:
            st.error(f"Failed to initialize AI engine: {str(e)}")
            return
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask about your chat..."):
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            # Create placeholder for status updates
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            try:
                # Step 1: Searching
                status_placeholder.info("🔍 Searching through your messages...")
                import time
                time.sleep(0.3)  # Brief pause for visual feedback
                
                # Step 2: Processing
                status_placeholder.info("🧠 Analyzing relevant conversations...")
                time.sleep(0.3)
                
                # Step 3: Generating
                status_placeholder.info("✨ Generating response...")
                
                # Query the AI with conversation history
                result = engine.query(prompt, chat_history=st.session_state.chat_history)
                answer = result['answer']
                sources = result['sources']
                
                # Clear status and show response
                status_placeholder.empty()
                response_placeholder.markdown(answer)
                
                # Show sources with enhanced information
                if sources:
                    with st.expander(f"📚 Sources - {len(sources)} relevant messages (Click to view)"):
                        for i, source in enumerate(sources, 1):
                            date_time = f"{source.get('date', '')} {source.get('time', '')}" if source.get('date') else source.get('timestamp', '')
                            st.markdown(f"**{i}. [{date_time}]**")
                            st.markdown(f"**{source.get('sender', 'Unknown')}:** {source['content']}")
                            st.divider()
                
                st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
            
            except Exception as e:
                status_placeholder.empty()
                error_msg = f"❌ Error: {str(e)}"
                response_placeholder.error(error_msg)
                st.caption("💡 Tip: Try rephrasing your question or check your API key.")


def show_emoji_words_tab():
    """
    Emoji and word frequency analysis tab.
    """
    st.header("😊 Emoji & Word Analysis")
    
    group_id = st.session_state.selected_group_id
    
    with DatabaseManager.get_session() as session:
        # Emoji Statistics
        st.subheader("🎭 Top Emojis")
        emoji_df = SQLAnalytics.get_emoji_statistics(session, group_id)
        
        if not emoji_df.empty:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(emoji_df, use_container_width=True, height=400)
            
            with col2:
                # Create emoji bar chart
                import plotly.express as px
                fig = px.bar(
                    emoji_df.head(15), 
                    x='count', 
                    y='emoji',
                    orientation='h',
                    title='Most Used Emojis',
                    labels={'count': 'Usage Count', 'emoji': 'Emoji'},
                    color='count',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emojis found in this chat")
        
        st.divider()
        
        # Word Frequency
        st.subheader("📝 Most Common Words")
        word_df = SQLAnalytics.get_word_frequency(session, group_id, top_n=50)
        
        if not word_df.empty:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(word_df.head(20), use_container_width=True, height=400)
            
            with col2:
                # Word cloud visualization
                try:
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt
                    
                    # Create word frequency dict
                    word_freq = dict(zip(word_df['word'], word_df['count']))
                    
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400,
                        background_color='white',
                        colormap='viridis',
                        relative_scaling=0.5,
                        min_font_size=10
                    ).generate_from_frequencies(word_freq)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Word Cloud', fontsize=16, pad=20)
                    st.pyplot(fig)
                except ImportError:
                    # Fallback to bar chart if wordcloud not available
                    import plotly.express as px
                    fig = px.bar(
                        word_df.head(20), 
                        x='count', 
                        y='word',
                        orientation='h',
                        title='Top 20 Words',
                        labels={'count': 'Frequency', 'word': 'Word'},
                        color='count',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No word data available")
        
        st.divider()
        
        # Message Length Statistics
        st.subheader("📏 Message Length Analysis")
        length_df = SQLAnalytics.get_message_length_stats(session, group_id)
        
        if not length_df.empty:
            import plotly.express as px
            
            fig = px.scatter(
                length_df,
                x='message_count',
                y='avg_length',
                size='max_length',
                hover_data=['sender'],
                title='Message Length vs Activity',
                labels={
                    'message_count': 'Total Messages',
                    'avg_length': 'Average Message Length (chars)',
                    'max_length': 'Longest Message'
                },
                color='avg_length',
                color_continuous_scale='Plasma'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(length_df, use_container_width=True)


def show_temporal_patterns_tab():
    """
    Temporal patterns and activity analysis tab.
    """
    st.header("📈 Temporal Activity Patterns")
    
    group_id = st.session_state.selected_group_id
    
    with DatabaseManager.get_session() as session:
        # Weekly Activity Pattern
        st.subheader("📅 Activity by Day of Week")
        weekly_df = SQLAnalytics.get_weekly_activity(session, group_id)
        
        if not weekly_df.empty:
            import plotly.express as px
            
            fig = px.bar(
                weekly_df,
                x='day_of_week',
                y='message_count',
                title='Messages by Day of Week',
                labels={'message_count': 'Message Count', 'day_of_week': 'Day'},
                color='message_count',
                color_continuous_scale='Sunset'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Daily Activity Timeline
        st.subheader("📆 Daily Activity Timeline")
        daily_df = SQLAnalytics.get_daily_activity(session, group_id)
        
        if not daily_df.empty:
            import plotly.express as px
            
            fig = px.line(
                daily_df,
                x='date',
                y='message_count',
                title='Message Activity Over Time',
                labels={'message_count': 'Messages', 'date': 'Date'},
                markers=True
            )
            fig.update_traces(line_color='#1f77b4', line_width=2)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            most_active_date = daily_df.loc[daily_df['message_count'].idxmax(), 'date']
            col1.metric("Most Active Day", str(most_active_date))
            col2.metric("Peak Messages", int(daily_df['message_count'].max()))
            col3.metric("Avg Messages/Day", int(daily_df['message_count'].mean()))
        
        st.divider()
        
        # Response Time Analysis
        st.subheader("⚡ Response Time Analysis")
        response_df = SQLAnalytics.get_response_time_analysis(session, group_id)
        
        if not response_df.empty:
            st.markdown("**Fastest Response Pairs** (within 1 hour)")
            
            # Format response time
            response_df['response_time_formatted'] = response_df['avg_response_time'].apply(
                lambda x: f"{int(x)} min" if x < 60 else f"{x/60:.1f} hr"
            )
            
            import plotly.express as px
            
            response_df['pair'] = response_df['from_user'] + ' → ' + response_df['to_user']
            
            fig = px.bar(
                response_df.head(10),
                x='avg_response_time',
                y='pair',
                orientation='h',
                title='Top 10 Fastest Response Pairs',
                labels={'avg_response_time': 'Avg Response Time (minutes)', 'pair': 'User Pair'},
                color='avg_response_time',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                response_df[['from_user', 'to_user', 'response_time_formatted']].rename(
                    columns={
                        'from_user': 'From',
                        'to_user': 'To',
                        'response_time_formatted': 'Avg Response Time'
                    }
                ),
                use_container_width=True
            )
        else:
            st.info("No response time data available")



def main():
    """
    Main application entry point.
    """
    if not st.session_state.logged_in:
        login_page()
    else:
        dashboard_page()


if __name__ == "__main__":
    main()