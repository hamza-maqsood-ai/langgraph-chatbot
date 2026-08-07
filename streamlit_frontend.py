import streamlit as st
import time
from typing import List, Dict, Any

# ----- YOUR BACKEND (unchanged) -----
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# -----------------------------------------------------------------------------
# FRONTEND – Premium UI with Enhanced Animations
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ===== Animated Background ===== */
        .stApp {
            background: linear-gradient(135deg, #0a0a1a, #1a1a3e, #0a0a1a);
            background-size: 400% 400%;
            animation: gradientMove 15s ease infinite;
            color: #f0f0f0;
        }
        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* ===== Glass Sidebar ===== */
        section[data-testid="stSidebar"] {
            background: rgba(20, 20, 35, 0.6) !important;
            backdrop-filter: blur(24px) !important;
            -webkit-backdrop-filter: blur(24px) !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
            box-shadow: 6px 0 40px rgba(0,0,0,0.4) !important;
        }

        .sidebar-content {
            padding: 1.8rem 1.5rem;
        }

        .sidebar-title {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #4f46e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }

        .sidebar-subtitle {
            font-size: 0.85rem;
            color: #a0a0b0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }

        /* ===== Buttons with scale + glow ===== */
        .stButton > button {
            width: 100%;
            background: rgba(255,255,255,0.05) !important;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 14px !important;
            color: #e0e0e0 !important;
            font-weight: 500 !important;
            padding: 0.7rem !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
        }
        .stButton > button:hover {
            background: rgba(255,255,255,0.12) !important;
            transform: scale(1.02) translateY(-3px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.4), 0 0 15px rgba(100,100,255,0.2) !important;
            border-color: rgba(255,255,255,0.2) !important;
        }
        .stButton > button:active {
            transform: scale(0.98);
        }

        /* ===== Stats ===== */
        .stat-box {
            background: rgba(255,255,255,0.04);
            border-radius: 16px;
            padding: 1.2rem;
            margin: 1.5rem 0;
            border: 1px solid rgba(255,255,255,0.06);
            backdrop-filter: blur(4px);
        }
        .stat-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9090a0;
        }
        .stat-value {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #4f46e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* ===== Model Badge ===== */
        .model-badge {
            display: inline-block;
            background: rgba(100,100,255,0.12);
            border-radius: 30px;
            padding: 0.3rem 1rem;
            font-size: 0.75rem;
            font-weight: 500;
            color: #b0b0ff;
            border: 1px solid rgba(100,100,255,0.2);
            backdrop-filter: blur(4px);
        }

        /* ===== About ===== */
        .about-box {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.04);
            font-size: 0.85rem;
            color: #b0b0c0;
            line-height: 1.6;
            backdrop-filter: blur(4px);
        }

        /* ===== Welcome Card ===== */
        .welcome-card {
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 32px;
            padding: 3.5rem 2.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 30px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
            margin: auto;
            max-width: 650px;
            width: 100%;
            animation: floatIn 0.8s ease;
        }
        @keyframes floatIn {
            0% { opacity: 0; transform: translateY(30px) scale(0.95); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        .welcome-icon {
            font-size: 4.5rem;
            display: inline-block;
            animation: bounce 3s infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-12px); }
        }
        .welcome-title {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #4f46e5, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 0.5rem;
        }
        .welcome-sub {
            color: #c0c0d0;
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        .welcome-hint {
            background: rgba(255,255,255,0.06);
            border-radius: 40px;
            padding: 0.6rem 1.8rem;
            display: inline-block;
            margin-top: 1.8rem;
            color: #b0b0c0;
            font-size: 0.9rem;
            border: 1px solid rgba(255,255,255,0.06);
            backdrop-filter: blur(4px);
        }

        /* ===== Chat Messages ===== */
        .chat-message {
            display: flex;
            margin-bottom: 1.2rem;
            animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes slideIn {
            0% { opacity: 0; transform: translateY(20px) scale(0.96); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }

        .message-content {
            max-width: 75%;
            padding: 0.9rem 1.6rem;
            border-radius: 24px;
            font-size: 0.95rem;
            line-height: 1.7;
            box-shadow: 0 4px 24px rgba(0,0,0,0.2);
            word-wrap: break-word;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .message-content:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }

        .user-message {
            justify-content: flex-end;
        }
        .user-message .message-content {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border-bottom-right-radius: 6px;
        }

        .assistant-message {
            justify-content: flex-start;
        }
        .assistant-message .message-content {
            background: rgba(40, 40, 55, 0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.05);
            color: #e8e8e8;
            border-bottom-left-radius: 6px;
        }

        /* ===== Typing Indicator (Bouncing Dots) ===== */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.6rem 1.5rem;
            background: rgba(40, 40, 55, 0.5);
            backdrop-filter: blur(8px);
            border-radius: 40px;
            border: 1px solid rgba(255,255,255,0.04);
            width: fit-content;
            margin-bottom: 1.2rem;
        }
        .typing-dots {
            display: flex;
            gap: 0.4rem;
        }
        .typing-dots span {
            width: 10px;
            height: 10px;
            background: #a78bfa;
            border-radius: 50%;
            animation: dotBounce 1.4s infinite both;
        }
        .typing-dots span:nth-child(1) { animation-delay: 0s; }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes dotBounce {
            0%, 80%, 100% { transform: scale(0.4); opacity: 0.4; }
            40% { transform: scale(1); opacity: 1; }
        }
        .typing-label {
            color: #c0c0d0;
            font-size: 0.9rem;
        }

        /* ===== Sticky Chat Input with Glow ===== */
        .stChatInputContainer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem 2rem 1.8rem 2rem;
            background: linear-gradient(to top, rgba(10,10,20,0.95), rgba(10,10,20,0));
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            z-index: 100;
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            justify-content: center;
        }
        .stChatInputContainer > div {
            width: 100%;
            max-width: 800px;
        }
        .stChatInputContainer textarea {
            background: rgba(30, 30, 45, 0.7) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 30px !important;
            color: #f0f0f0 !important;
            padding: 0.8rem 1.8rem !important;
            font-size: 0.95rem !important;
            box-shadow: 0 8px 30px rgba(0,0,0,0.5) !important;
            transition: all 0.3s ease !important;
        }
        .stChatInputContainer textarea:focus {
            border-color: rgba(100, 100, 255, 0.4) !important;
            box-shadow: 0 8px 40px rgba(0,0,0,0.6), 0 0 0 4px rgba(100,100,255,0.1), 0 0 30px rgba(100,100,255,0.15) !important;
        }

        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

        /* ===== Responsive ===== */
        @media (max-width: 768px) {
            .chat-container { padding: 0.5rem 1rem 5rem 1rem; }
            .stChatInputContainer { padding: 0.5rem 1rem 1.2rem 1rem; }
            .message-content { max-width: 85%; font-size: 0.9rem; padding: 0.7rem 1.2rem; }
            .welcome-card { padding: 2rem 1.5rem; }
            .welcome-title { font-size: 1.8rem; }
        }
    </style>
    """

st.markdown(load_css(), unsafe_allow_html=True)

# ----- Sidebar -----
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🤖 LangGraph AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Powered by LangGraph + Groq + Streamlit</div>', unsafe_allow_html=True)

    if st.button("✨ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.config = {"configurable": {"thread_id": str(time.time())}}
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    msg_count = len(st.session_state.get("messages", []))
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Conversation</div>
        <div class="stat-value">{msg_count}</div>
        <div style="font-size:0.8rem; color:#808090;">messages exchanged</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;"><span style="color:#9090a0; font-size:0.8rem;">Current Model</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="model-badge">llama-3.3-70b-versatile (Groq)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box" style="margin-top:2rem;">
        <strong style="color:#d0d0e0;">About</strong><br>
        This assistant uses LangGraph for stateful conversations, Groq's ultra‑fast inference, and a sleek glass‑morphism UI. Built with ❤️.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----- Main Chat Area -----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": str(time.time())}}

chat_container = st.container()

# Welcome card
if len(st.session_state.messages) == 0:
    with chat_container:
        st.markdown("""
        <div style="display:flex; justify-content:center; align-items:center; min-height:70vh;">
            <div class="welcome-card">
                <div class="welcome-icon">🤖</div>
                <div class="welcome-title">LangGraph AI Assistant</div>
                <div class="welcome-sub">How can I help you today?</div>
                <div class="welcome-hint">Ask me anything — I'm powered by Groq's llama-3.3</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Display messages
else:
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-content">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-content">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

typing_placeholder = st.empty()

# ----- Chat Input -----
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# Generate response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with typing_placeholder:
        st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
            <span class="typing-label">Assistant is thinking</span>
        </div>
        """, unsafe_allow_html=True)

    # Convert to LangChain messages
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))

    try:
        result = chatbot.invoke({"messages": langchain_messages}, st.session_state.config)
        assistant_response = result["messages"][-1].content
    except Exception as e:
        assistant_response = f"⚠️ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    typing_placeholder.empty()
    st.rerun()

# Auto‑scroll
scroll_script = """
<script>
    function scrollToBottom() {
        const main = document.querySelector('.main');
        if (main) main.scrollTop = main.scrollHeight;
        window.scrollTo(0, document.body.scrollHeight);
    }
    setTimeout(scrollToBottom, 100);
</script>
"""
st.components.v1.html(scroll_script, height=0, width=0)