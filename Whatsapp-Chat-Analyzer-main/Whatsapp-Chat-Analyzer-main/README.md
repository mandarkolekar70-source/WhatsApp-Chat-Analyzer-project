# 📱 SocialPulse | WhatsApp Chat Analyzer 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜🔗_LangChain-0.1.0-green?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq_AI-Fast-orange?style=for-the-badge)

> **Unlock the hidden stories in your WhatsApp chats with AI-powered insights.**

SocialPulse is a production-grade analytics platform that transforms your raw chat exports into beautiful, interactive dashboards. It uses **Graph Theory** to map social connections and **Generative AI (RAG)** to let you chat with your data.

---

## 🌟 Key Features

### 📊 **Deep Analytics Dashboard**
- **Activity Heatmaps:** Visualize when your group is most active (hourly/weekly).
- **Emoji Analysis:** Discover the most used reactions and emotions.
- **Word Clouds:** See the topics that dominate the conversation.
- **User Rankings:** Identify top contributors and chatterboxes.

### 🕸️ **Social Network Graph**
- **Influence Mapping:** Uses **PageRank** algorithms to determine who drives the conversation.
- **Interactive Visualization:** Drag, zoom, and explore connections between members.
- **Reply Heuristics:** Smart detection of "who replies to whom" (based on 120s time windows).

### 🤖 **AI Chat Assistant (RAG)**
- **Ask Anything:** "What did we decide for the trip?", "Who mentioned the deadline?", "Summarize last week's drama."
- **Powered by Groq:** Lightning-fast inference using **Llama 3.3 70B**.
- **Context Aware:** Remembers conversation history for natural follow-ups.
- **Source Citations:** Shows you exactly which messages were used to generate the answer.

### 🔐 **Enterprise-Grade Security**
- **User Authentication:** Secure signup/login with encrypted passwords (bcrypt).
- **Private Data:** Your chats are processed locally and stored securely in your SQL database.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Rapid interactive UI development |
| **Database** | PostgreSQL | Robust relational data storage |
| **ORM** | SQLAlchemy | Pythonic database interactions |
| **AI / LLM** | Groq (Llama 3) | Ultra-fast inference engine |
| **Orchestration** | LangChain | RAG pipeline management |
| **Vector Store** | FAISS | Efficient similarity search for chat messages |
| **Graph Theory** | NetworkX | Social network analysis and metrics |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/AkshayPatil20001/Whatsapp-Chat-Analyzer.git
cd Whatsapp-Chat-Analyzer
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Secrets
Create a `.env` file in the root directory:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/socialpulse
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the App
```bash
streamlit run app.py
```
👉 **Open [http://localhost:8501](http://localhost:8501) in your browser.**

---

## 📸 Screenshots

| **Dashboard Overview** | **Social Graph** |
|:---:|:---:|
| * Visualize activity trends and top users * | * Explore who influences whom in the group * |
| ![Dashboard](assets/dashboard.PNG) | ![Social Graph](assets/social_graph.PNG) |

| **AI Chat Assistant** | **Emoji Analysis** |
|:---:|:---:|
| * Chat with your data using natural language * | * See the emotional pulse of the chat * |
| ![AI Chat](assets/ai_chat.PNG) | ![Emoji Analysis](assets/emoji_analysis.PNG) |

---

## 💡 How It Works (Under the Hood)

1.  **Parsing:** The custom parsers handle various WhatsApp date formats and handle multi-line messages effortlessly.
2.  **Vectorization:** Messages are embedded using `all-MiniLM-L6-v2` and stored in a local **FAISS** index for semantic search.
3.  **Graph Construction:** We build a directed graph where an edge `A -> B` is created if User B replies to User A within 2 minutes.
4.  **Retrieval Augmented Generation (RAG):** When you ask a question, the system retrieves relevant chat snippets and feeds them to the **Groq Llama 3** model to generate a precise answer.

---

<div align="center">
  <sub>Built with ❤️ using Python and Streamlit</sub>
</div>