import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()  # reads GROQ_API_KEY from your existing .env file

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
BACKEND_READY = bool(GROQ_API_KEY)

if BACKEND_READY:

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.7,
    )

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS & STYLING - CONSISTENT LIGHT THEME EVERYWHERE
# ============================================================================
custom_css = """
<style>
    * {
        box-sizing: border-box;
    }

    :root {
        --primary-color: #2563eb;
        --primary-dark: #1e40af;
        --primary-light: #3b82f6;
        --secondary-color: #7c3aed;
        --background-light: #f8fafc;
        --surface-light: #ffffff;
        --surface-hover: #f0f4f8;
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --border-soft: rgba(37, 99, 235, 0.15);
        --success-color: #10b981;
        --danger-color: #dc2626;
    }

    /* ============ GLOBAL LIGHT BACKGROUND (every Streamlit container) ============ */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 35%, #f5f3ff 100%) !important;
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif !important;
    }

    [data-testid="stHeader"] { box-shadow: none !important; }

    [data-testid="stBottom"] > div {
        background: transparent !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
        max-width: 900px !important;
    }

    /* ============ SIDEBAR ============ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f5f7ff 100%) !important;
        border-right: 1px solid var(--border-soft);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebarContent"] { padding-top: 1rem; }

    /* ============ HEADER (softer, not heavy) ============ */
    .header-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(238,242,255,0.9) 100%);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.08);
        animation: fadeSlideDown 0.6s ease-out;
    }

    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .header-title {
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
        background: linear-gradient(90deg, #2563eb, #7c3aed 60%, #2563eb);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: shineText 6s linear infinite;
    }

    @keyframes shineText {
        to { background-position: 200% center; }
    }

    .header-subtitle {
        margin: 0.35rem 0 0 0;
        font-size: 0.92rem;
        font-weight: 400;
        color: var(--text-muted);
    }

    /* ============ SIDEBAR BUTTONS ============ */
    .stButton > button {
        width: 100%;
        padding: 0.65rem 1rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: none !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stSidebarContent"] div[data-testid="column"]:nth-of-type(1) .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    }
    [data-testid="stSidebarContent"] div[data-testid="column"]:nth-of-type(1) .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4) !important;
    }

    [data-testid="stSidebarContent"] div[data-testid="column"]:nth-of-type(2) .stButton > button {
        background: rgba(220, 38, 38, 0.08) !important;
        color: var(--danger-color) !important;
        border: 1.5px solid rgba(220, 38, 38, 0.25) !important;
    }
    [data-testid="stSidebarContent"] div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {
        background: rgba(220, 38, 38, 0.15) !important;
        transform: translateY(-2px);
    }

    /* History item buttons */
    .history-btn button {
        background: #ffffff !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-soft) !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-weight: 500 !important;
        padding: 0.5rem 0.75rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .history-btn button:hover {
        background: rgba(37, 99, 235, 0.08) !important;
        border-color: var(--primary-color) !important;
        color: var(--primary-dark) !important;
        transform: translateX(3px);
    }

    .sidebar-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--primary-dark) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.25rem 0 0.6rem 0;
        opacity: 0.85;
    }

    .sidebar-stat {
        padding: 0.85rem 1rem;
        background: linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(124,58,237,0.04) 100%);
        border: 1px solid var(--border-soft);
        border-radius: 10px;
        margin-bottom: 0.6rem;
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; } to { opacity: 1; }
    }

    .stat-label {
        font-size: 0.72rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--primary-dark) !important;
        margin-top: 0.15rem;
    }

    .about-section {
        padding: 1rem;
        background: #ffffff;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        font-size: 0.85rem;
    }

    hr, [data-testid="stSidebar"] hr {
        border-color: var(--border-soft) !important;
        margin: 1rem 0 !important;
    }

    /* ============ WELCOME CARD ============ */
    .welcome-container {
        display: flex;
        justify-content: center;
        padding: 2rem 0;
    }

    .welcome-card {
        background: #ffffff;
        border: 1px solid var(--border-soft);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        max-width: 640px;
        width: 100%;
        text-align: center;
        box-shadow: 0 10px 40px rgba(37, 99, 235, 0.1);
        animation: cardPopIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    @keyframes cardPopIn {
        from { opacity: 0; transform: scale(0.94) translateY(10px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }

    .welcome-icon {
        font-size: 3.2rem;
        margin-bottom: 0.75rem;
        display: inline-block;
        animation: floatIcon 3s ease-in-out infinite;
    }

    @keyframes floatIcon {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-8px) rotate(-4deg); }
    }

    .welcome-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.6rem 0;
    }

    .welcome-subtitle {
        font-size: 0.95rem;
        color: var(--text-muted);
        line-height: 1.6;
        max-width: 460px;
        margin: 0 auto 1.75rem auto;
    }

    .welcome-features {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
    }

    .feature-item {
        background: linear-gradient(135deg, rgba(37,99,235,0.06), rgba(124,58,237,0.05));
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 0.85rem;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-secondary);
        transition: all 0.25s ease;
    }

    .feature-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.15);
        border-color: var(--primary-color);
        color: var(--primary-dark);
    }

    /* ============ CHAT MESSAGES ============ */
    .chat-message {
        display: flex;
        margin-bottom: 1.1rem;
    }

    .chat-message.user { justify-content: flex-end; }
    .chat-message.assistant { justify-content: flex-start; }

    .user-message {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: #ffffff !important;
        padding: 0.9rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 72%;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
        animation: slideInRight 0.35s ease-out;
        line-height: 1.55;
        font-size: 0.94rem;
    }

    .user-message .message-text, .user-message .message-time {
        color: #ffffff !important;
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(16px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .assistant-message {
        background: #ffffff;
        border: 1px solid var(--border-soft);
        color: var(--text-primary) !important;
        padding: 0.9rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 72%;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
        animation: slideInLeft 0.35s ease-out;
        line-height: 1.6;
        font-size: 0.94rem;
    }

    .assistant-message .message-text, .assistant-message .message-time {
        color: var(--text-primary) !important;
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-16px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .message-text { word-wrap: break-word; }
    .message-text p { margin: 0.4rem 0; }
    .message-text p:first-child { margin-top: 0; }
    .message-text p:last-child { margin-bottom: 0; }

    .message-time {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 0.4rem;
        font-weight: 500;
    }

    .typing-indicator {
        display: flex;
        gap: 0.4rem;
        padding: 0.4rem 0.2rem;
    }

    .typing-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--primary-color);
        animation: typingBounce 1.3s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.15s; }
    .typing-dot:nth-child(3) { animation-delay: 0.3s; }

    @keyframes typingBounce {
        0%, 60%, 100% { opacity: 0.4; transform: translateY(0); }
        30% { opacity: 1; transform: translateY(-6px); }
    }

    /* ============ CHAT INPUT (modern Streamlit selectors) ============ */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1.5px solid var(--border-soft) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08) !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12), 0 6px 18px rgba(37, 99, 235, 0.15) !important;
    }

    [data-testid="stChatInputTextArea"],
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stChatInputTextArea"]::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.8;
    }

    [data-testid="stChatInputSubmitButton"] {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInputSubmitButton"] svg { fill: #ffffff !important; }

    /* Legacy selectors kept as a fallback for older Streamlit builds */
    .stChatInputContainer, .stChatFloatingInputContainer {
        background: transparent !important;
    }
    .stChatInputContainer input {
        background: #ffffff !important;
        color: var(--text-primary) !important;
        border: 1.5px solid var(--border-soft) !important;
        border-radius: 16px !important;
    }

    /* ============ MARKDOWN / MISC ============ */
    .stMarkdown, .stMarkdown p { color: var(--text-primary) !important; }
    .stMarkdown a { color: var(--primary-dark) !important; font-weight: 600; }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(37, 99, 235, 0.25);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(37, 99, 235, 0.4); }

    @media (max-width: 768px) {
        .header-title { font-size: 1.4rem; }
        .user-message, .assistant-message { max-width: 92%; }
        .welcome-features { grid-template-columns: 1fr; }
        .welcome-card { padding: 2rem 1.25rem; }
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "conversations" not in st.session_state:
    # Each item: {"id": str, "title": str, "messages": [...], "created": str}
    st.session_state.conversations = []

if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()

if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False


def _archive_current_chat():
    """Save the active conversation into history before starting a new one."""
    if st.session_state.message_history:
        first_user_msg = next(
            (m["content"] for m in st.session_state.message_history if m["role"] == "user"),
            "Conversation",
        )
        title = (first_user_msg[:38] + "…") if len(first_user_msg) > 38 else first_user_msg
        st.session_state.conversations.insert(
            0,
            {
                "id": str(uuid.uuid4()),
                "title": title,
                "messages": st.session_state.message_history.copy(),
                "created": datetime.now().strftime("%b %d, %H:%M"),
            },
        )


# ============================================================================
# GROQ API BACKEND INTEGRATION (unchanged logic)
# ============================================================================
def invoke_chatbot(user_message: str) -> str:
    """
    Call your LangGraph chatbot with Groq API
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            return "❌ Error: GROQ_API_KEY not found in .env file. Please add it and restart the app."

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=groq_api_key,
        )

        system_prompt = """You are LangGraph AI, a helpful, intelligent, and respectful AI assistant.

Your characteristics:
- You provide accurate, well-researched information
- You're engaging and friendly in conversation
- You think step-by-step for complex problems
- You're honest about what you don't know
- You format responses clearly with good structure
- You keep responses concise but comprehensive

Language rule (very important):
- Always reply in the SAME language and style the user just used.
- If the user writes in Roman Urdu (Urdu written using English/Latin letters, e.g. "aap kaisy hain"), you MUST reply in Roman Urdu too — not in English and not in Urdu script.
- If the user writes in English, reply in English. If they mix English and Roman Urdu, mirror that mix naturally.
- Never switch language on your own; always match the user's most recent message.

When appropriate, use markdown formatting for lists, code blocks, and emphasis."""

        messages = [SystemMessage(content=system_prompt)]

        for msg in st.session_state.message_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_message))

        response = llm.invoke(messages)
        return response.content

    except ImportError:
        return "❌ Error: langchain_groq not installed. Run: pip install langchain-groq"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# HEADER
# ============================================================================
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🤖 LangGraph AI Assistant</h1>
        <p class="header-subtitle">Powered by LangGraph + Groq + Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 💬 Chat Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            _archive_current_chat()
            st.session_state.message_history = []
            st.session_state.conversation_count += 1
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.message_history = []
            st.rerun()

    st.divider()

    # Statistics
    st.markdown('<p class="sidebar-section-title">📊 Statistics</p>', unsafe_allow_html=True)

    scol1, scol2 = st.columns(2)
    with scol1:
        st.markdown(
            f"""
            <div class="sidebar-stat">
                <div class="stat-label">Messages</div>
                <div class="stat-value">{len(st.session_state.message_history)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with scol2:
        st.markdown(
            f"""
            <div class="sidebar-stat">
                <div class="stat-label">Conversations</div>
                <div class="stat-value">{len(st.session_state.conversations)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elapsed_time = datetime.now() - st.session_state.start_time
    minutes = int(elapsed_time.total_seconds() / 60)

    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div class="stat-label">Session Time</div>
            <div class="stat-value">{minutes}m</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Chat History ----
    st.markdown('<p class="sidebar-section-title">🕘 Recent Conversations</p>', unsafe_allow_html=True)

    if not st.session_state.conversations:
        st.markdown(
            '<p style="font-size:0.82rem; color:#64748b;">No saved conversations yet. '
            'Start chatting, then hit "New Chat" to save this one.</p>',
            unsafe_allow_html=True,
        )
    else:
        for convo in st.session_state.conversations[:12]:
            st.markdown('<div class="history-btn">', unsafe_allow_html=True)
            if st.button(f"💬 {convo['title']}", key=f"history_{convo['id']}", use_container_width=True):
                _archive_current_chat()
                st.session_state.message_history = convo["messages"].copy()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Model Information
    st.markdown('<p class="sidebar-section-title">⚙️ Configuration</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-stat">
            <div class="stat-label">Current Model</div>
            <div style="color: #2563eb; font-weight: 700; margin-top: 0.4rem; font-size: 0.95rem;">
                llama-3.3-70b
            </div>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 0.2rem; font-weight: 600;">
                Provider: Groq
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # About Section
    st.markdown('<p class="sidebar-section-title">ℹ️ About</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="about-section">
            <div style="font-weight:700; color:#1e293b; margin-bottom:0.4rem;">LangGraph AI</div>
            <p style="margin: 0 0 0.75rem 0; color: #475569;">
                A modern, premium chatbot interface powered by LangGraph, Groq API, and Streamlit.
            </p>
            <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(37, 99, 235, 0.15);">
                <p style="margin: 0; font-size: 0.82rem; color: #64748b; line-height: 1.75; font-weight: 500;">
                    <strong style="color: #2563eb;">Features:</strong><br>
                    ✨ Real-time responses<br>
                    🎨 Modern UI/UX<br>
                    ⚡ Fast inference<br>
                    🔒 Secure API<br>
                    💬 Multi-turn conversations
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# MAIN CHAT AREA
# ============================================================================
if not st.session_state.message_history:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-card">
                <div class="welcome-icon">🚀</div>
                <h2 class="welcome-title">Welcome to LangGraph AI</h2>
                <p class="welcome-subtitle">
                    Start a conversation with our advanced AI assistant powered by LangGraph and Groq's latest models.
                </p>
                <div class="welcome-features">
                    <div class="feature-item">⚡ Lightning Fast</div>
                    <div class="feature-item">🤖 Advanced AI</div>
                    <div class="feature-item">💬 Natural Language</div>
                    <div class="feature-item">🎯 Accurate Responses</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for message in st.session_state.message_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")

        if role == "user":
            st.markdown(
                f"""
                <div class="chat-message user">
                    <div class="user-message">
                        <div class="message-text">{content}</div>
                        <div class="message-time">{timestamp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-message assistant">
                    <div class="assistant-message">
                        <div class="message-text">{content}</div>
                        <div class="message-time">{timestamp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================================
# CHAT INPUT
# ============================================================================
user_input = st.chat_input(
    "💬 Type your message here...",
    key="user_input",
)

# ============================================================================
# MESSAGE HANDLING & RESPONSE GENERATION
# ============================================================================
if user_input:
    user_timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": user_timestamp,
    })
    st.session_state.waiting_for_response = True
    st.rerun()

if st.session_state.waiting_for_response:
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        """
        <div class="chat-message assistant">
            <div class="assistant-message">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        response_text = invoke_chatbot(st.session_state.message_history[-1]["content"])
        typing_placeholder.empty()

        assistant_timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.message_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": assistant_timestamp,
        })

        st.session_state.waiting_for_response = False
        st.rerun()

    except Exception as e:
        typing_placeholder.empty()
        st.error(f"❌ Error: {str(e)}")
        if st.session_state.message_history and st.session_state.message_history[-1]["role"] == "user":
            st.session_state.message_history.pop()
        st.session_state.waiting_for_response = False

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(
    """
    <hr style="border: 1px solid rgba(37, 99, 235, 0.15); margin: 2rem 0 0 0;">
    <p style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 1rem 0; margin: 0;">
        Made with ❤️ using Streamlit • Powered by LangGraph & Groq API
    </p>
    """,
    unsafe_allow_html=True,
)
