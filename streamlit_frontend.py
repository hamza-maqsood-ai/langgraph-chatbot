"""
LangGraph AI Assistant — Premium Chat UI
Frontend : Streamlit (custom glassmorphism theme)
Backend  : LangGraph + Groq (llama-3.3-70b-versatile)

This file is split into two clearly marked zones:

  1. BACKEND  — the LangGraph / ChatGroq wiring. Kept in the exact
     shape the brief specified (CONFIG, chatbot.invoke, message_history,
     st.session_state, ChatGroq, llama-3.3-70b-versatile). If your real
     backend differs even slightly, just paste it in over this block —
     nothing below depends on its internals, only on the names
     `chatbot`, `CONFIG` and `st.session_state.message_history`.

  2. FRONTEND — everything visual. Dark glass theme, gradient bubbles,
     sidebar, welcome screen, typing indicator, sticky input.
"""

import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

# ============================================================================
# 1. BACKEND — LangGraph + Groq   (DO NOT MODIFY IN PRODUCTION)
# ----------------------------------------------------------------------------
# Swap this block for your existing implementation if it differs — the
# frontend only ever calls `chatbot.invoke(..., config=CONFIG)` and reads
# `st.session_state.message_history`, so naming compatibility is all that
# matters.
# ============================================================================

load_dotenv()  # reads GROQ_API_KEY from your existing .env file

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
BACKEND_READY = bool(GROQ_API_KEY)

if BACKEND_READY:

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.7,
    )

    def chat_node(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    checkpointer = MemorySaver()
    chatbot = graph.compile(checkpointer=checkpointer)

# ============================================================================
# END BACKEND
# ============================================================================


# ============================================================================
# 2. FRONTEND
# ============================================================================

MODEL_LABEL = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state.message_history = []  # list[{"role", "content", "time"}]

CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}


def start_new_conversation():
    """Used by both 'New Chat' and 'Clear Chat' — resets the visible
    history *and* the LangGraph memory thread together, so the UI and
    the model's actual context never drift apart."""
    st.session_state.message_history = []
    st.session_state.thread_id = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Theme — fonts, palette, glassmorphism, motion
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg-deep:#0a0e1a;
  --bg-mid:#0f1428;
  --bg-violet:#1a1440;
  --glass-bg:rgba(255,255,255,0.045);
  --glass-bg-strong:rgba(255,255,255,0.075);
  --glass-border:rgba(255,255,255,0.09);
  --text-primary:#edeff7;
  --text-secondary:#9aa3bf;
  --text-tertiary:#6b7390;
  --user-a:#3b6ff6;
  --user-b:#8b5cf6;
  --assistant-bg:rgba(28,31,44,0.75);
  --accent-cyan:#22d3ee;
  --accent-glow:rgba(99,102,241,0.4);
  --shadow-soft:0 8px 28px rgba(0,0,0,0.35);
  --shadow-glow:0 0 36px rgba(99,102,241,0.22);
  --r-lg:22px;
  --r-md:16px;
  --r-sm:10px;
}

html,body,[data-testid="stAppViewContainer"]{
  font-family:'Inter',sans-serif;
  color:var(--text-primary);
}

/* ---------- animated aurora background ---------- */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 600px at 12% -10%, rgba(59,111,246,0.16), transparent 60%),
    radial-gradient(900px 700px at 110% 10%, rgba(139,92,246,0.18), transparent 55%),
    radial-gradient(1000px 800px at 50% 120%, rgba(34,211,238,0.08), transparent 60%),
    linear-gradient(160deg, var(--bg-deep) 0%, var(--bg-mid) 45%, var(--bg-violet) 100%);
  background-attachment:fixed;
}
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
html{ scroll-behavior:smooth; }

/* ---------- scrollbar ---------- */
::-webkit-scrollbar{ width:8px; height:8px; }
::-webkit-scrollbar-thumb{ background:rgba(255,255,255,0.14); border-radius:8px; }
::-webkit-scrollbar-thumb:hover{ background:rgba(255,255,255,0.24); }

/* ---------- glass utility ---------- */
.glass-card{
  background:var(--glass-bg);
  backdrop-filter:blur(18px);
  -webkit-backdrop-filter:blur(18px);
  border:1px solid var(--glass-border);
  border-radius:var(--r-lg);
  box-shadow:var(--shadow-soft);
}

/* ---------- header ---------- */
.app-header{
  text-align:center;
  padding:26px 20px 22px 20px;
  margin-bottom:22px;
  position:relative;
}
.app-title{
  font-family:'Sora',sans-serif;
  font-weight:800;
  font-size:2.1rem;
  letter-spacing:-0.02em;
  margin:0;
  background:linear-gradient(90deg,#8fb4ff 0%, #b39dff 45%, #7ce8f7 100%);
  -webkit-background-clip:text;
  background-clip:text;
  color:transparent;
}
.app-subtitle{
  font-family:'JetBrains Mono',monospace;
  font-size:0.82rem;
  color:var(--text-secondary);
  margin-top:8px;
  letter-spacing:0.02em;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(15,18,32,0.92), rgba(20,15,40,0.92));
  border-right:1px solid var(--glass-border);
  backdrop-filter:blur(18px);
}
section[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }
.sidebar-brand{
  font-family:'Sora',sans-serif;
  font-weight:700;
  font-size:1.05rem;
  color:var(--text-primary);
  margin-bottom:2px;
}
.sidebar-section-label{
  font-family:'JetBrains Mono',monospace;
  font-size:0.72rem;
  text-transform:uppercase;
  letter-spacing:0.08em;
  color:var(--text-tertiary);
  margin:18px 0 8px 2px;
}

.model-card{
  background:var(--glass-bg-strong);
  border:1px solid var(--glass-border);
  border-radius:var(--r-md);
  padding:12px 14px;
  margin-bottom:6px;
}
.model-card .row{ display:flex; align-items:center; gap:8px; }
.status-dot{
  width:8px; height:8px; border-radius:50%;
  background:#34d399;
  box-shadow:0 0 8px #34d399;
  flex-shrink:0;
  animation:pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot{ 0%,100%{opacity:1;} 50%{opacity:0.45;} }
.model-name{ font-family:'JetBrains Mono',monospace; font-size:0.86rem; color:var(--text-primary); }
.model-sub{ font-size:0.76rem; color:var(--text-tertiary); margin-top:2px; }

.stat-grid{ display:flex; gap:8px; }
.stat-chip{
  flex:1;
  background:var(--glass-bg-strong);
  border:1px solid var(--glass-border);
  border-radius:var(--r-sm);
  padding:10px 8px;
  text-align:center;
}
.stat-chip .num{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem; color:var(--text-primary); }
.stat-chip .lbl{ font-size:0.68rem; color:var(--text-tertiary); margin-top:2px; }

/* sidebar buttons */
section[data-testid="stSidebar"] .stButton>button{
  width:100%;
  border-radius:var(--r-md);
  border:1px solid var(--glass-border);
  background:var(--glass-bg-strong);
  color:var(--text-primary);
  font-family:'Inter',sans-serif;
  font-weight:600;
  padding:0.55rem 0.8rem;
  transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
section[data-testid="stSidebar"] .stButton>button:hover{
  transform:translateY(-2px);
  border-color:rgba(139,92,246,0.5);
  box-shadow:var(--shadow-glow);
}
section[data-testid="stSidebar"] .stButton>button:active{ transform:translateY(0px); }

/* primary New Chat button */
div[data-testid="stSidebarUserContent"] div.element-container:has(button[kind="primary"]) button{
  background:linear-gradient(135deg, var(--user-a), var(--user-b));
  border:none;
  color:#fff;
}

/* ---------- welcome card ---------- */
.welcome-wrap{ display:flex; justify-content:center; margin-top:2.5vh; }
.welcome-card{
  max-width:640px;
  width:100%;
  padding:38px 34px;
  text-align:center;
  animation:fadeInUp .5s ease;
}
.welcome-emoji{
  font-size:2.6rem;
  filter:drop-shadow(0 0 22px rgba(139,92,246,0.5));
}
.welcome-title{
  font-family:'Sora',sans-serif;
  font-weight:700;
  font-size:1.5rem;
  margin:12px 0 6px 0;
}
.welcome-text{
  color:var(--text-secondary);
  font-size:0.95rem;
  line-height:1.55;
  margin-bottom:6px;
}

/* suggestion buttons in main area */
div[data-testid="stMainBlockContainer"] div[data-testid="column"] .stButton>button{
  width:100%;
  text-align:left;
  border-radius:var(--r-md);
  border:1px solid var(--glass-border);
  background:var(--glass-bg-strong);
  color:var(--text-primary);
  padding:0.7rem 0.9rem;
  font-size:0.86rem;
  transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div[data-testid="stMainBlockContainer"] div[data-testid="column"] .stButton>button:hover{
  transform:translateY(-3px);
  border-color:rgba(34,211,238,0.45);
  box-shadow:0 10px 30px rgba(34,211,238,0.15);
}

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"]{
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
  padding:6px 0 !important;
  gap:10px;
  animation:fadeInUp .35s ease;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  flex-direction:row-reverse;
}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{
  border-radius:50% !important;
  box-shadow:0 0 0 2px rgba(255,255,255,0.08), 0 0 18px rgba(99,102,241,0.25);
}
[data-testid="stChatMessageContent"]{
  max-width:70%;
  width:auto !important;
  flex:0 1 auto !important;
  padding:13px 17px;
  box-shadow:var(--shadow-soft);
  transition:transform .18s ease, box-shadow .18s ease;
  font-size:0.94rem;
  line-height:1.6;
}
[data-testid="stChatMessageContent"]:hover{
  transform:translateY(-2px);
}
[data-testid="stChatMessageContent"] p{ margin-bottom:0.4em; }
[data-testid="stChatMessageContent"] p:last-child{ margin-bottom:0; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  background:linear-gradient(135deg, var(--user-a), var(--user-b));
  color:#ffffff;
  border-radius:20px 20px 4px 20px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"]{
  background:var(--assistant-bg);
  backdrop-filter:blur(14px);
  border:1px solid var(--glass-border);
  color:var(--text-primary);
  border-radius:20px 20px 20px 4px;
}
.msg-time{
  font-family:'JetBrains Mono',monospace;
  font-size:0.68rem;
  color:var(--text-tertiary);
  margin-top:5px;
  opacity:0.8;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .msg-time{ text-align:right; color:rgba(255,255,255,0.65); }

/* ---------- typing indicator ---------- */
.typing-row{ display:flex; align-items:center; gap:10px; padding:6px 0; animation:fadeInUp .3s ease; }
.typing-avatar{
  width:2rem;height:2rem;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  background:var(--assistant-bg);
  border:1px solid var(--glass-border);
  font-size:1rem;
}
.typing-bubble{
  background:var(--assistant-bg);
  backdrop-filter:blur(14px);
  border:1px solid var(--glass-border);
  border-radius:20px 20px 20px 4px;
  padding:14px 18px;
  display:flex;
  gap:5px;
  box-shadow:var(--shadow-soft);
}
.typing-bubble span{
  width:7px;height:7px;border-radius:50%;
  background:var(--accent-cyan);
  opacity:0.5;
  animation:bounce-dot 1.2s infinite ease-in-out;
}
.typing-bubble span:nth-child(2){ animation-delay:0.15s; }
.typing-bubble span:nth-child(3){ animation-delay:0.3s; }
@keyframes bounce-dot{
  0%,60%,100%{ transform:translateY(0); opacity:0.5; }
  30%{ transform:translateY(-6px); opacity:1; }
}

/* ---------- chat input ---------- */
[data-testid="stBottomBlockContainer"]{
  background:linear-gradient(180deg, transparent, rgba(10,14,26,0.85) 40%);
  backdrop-filter:blur(6px);
}
[data-testid="stChatInput"]{
  background:var(--glass-bg-strong) !important;
  border:1px solid var(--glass-border) !important;
  border-radius:26px !important;
  box-shadow:var(--shadow-soft);
  transition:box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stChatInput"]:focus-within{
  border-color:rgba(139,92,246,0.55) !important;
  box-shadow:0 0 0 4px rgba(139,92,246,0.14), var(--shadow-glow);
}
[data-testid="stChatInput"] textarea{
  color:var(--text-primary) !important;
  font-family:'Inter',sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder{ color:var(--text-tertiary) !important; }

/* ---------- misc ---------- */
.stAlert{ border-radius:var(--r-md); }
[data-testid="stExpander"]{
  background:var(--glass-bg);
  border:1px solid var(--glass-border);
  border-radius:var(--r-md);
}

@keyframes fadeInUp{
  from{ opacity:0; transform:translateY(10px); }
  to{ opacity:1; transform:translateY(0); }
}

/* ---------- responsive ---------- */
@media (max-width: 700px){
  [data-testid="stChatMessageContent"]{ max-width:86%; }
  .app-title{ font-size:1.6rem; }
  .welcome-card{ padding:26px 20px; }
}

/* ---------- reduced motion ---------- */
@media (prefers-reduced-motion: reduce){
  *{ animation:none !important; transition:none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🤖 LangGraph Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--text-tertiary);font-size:0.78rem;">AI workspace</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
    if st.button("🆕  New Chat", use_container_width=True, type="primary"):
        start_new_conversation()
        st.rerun()
    if st.button("🧹  Clear Chat", use_container_width=True):
        start_new_conversation()
        st.rerun()

    st.markdown('<div class="sidebar-section-label">Conversation Stats</div>', unsafe_allow_html=True)
    total_msgs = len(st.session_state.message_history)
    user_msgs = sum(1 for m in st.session_state.message_history if m["role"] == "user")
    ai_msgs = total_msgs - user_msgs
    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-chip"><div class="num">{total_msgs}</div><div class="lbl">Total</div></div>
          <div class="stat-chip"><div class="num">{user_msgs}</div><div class="lbl">You</div></div>
          <div class="stat-chip"><div class="num">{ai_msgs}</div><div class="lbl">AI</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Current Model</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="model-card">
          <div class="row">
            <span class="status-dot"></span>
            <span class="model-name">{MODEL_LABEL}</span>
          </div>
          <div class="model-sub">via Groq API &middot; LPU inference</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">About</div>', unsafe_allow_html=True)
    with st.expander("ℹ️  About this assistant"):
        st.markdown(
            """
            **LangGraph AI Assistant** is a conversational agent built on a
            LangGraph state graph, running inference through Groq's LPU
            infrastructure for near-instant responses.

            - **Orchestration:** LangGraph
            - **Inference:** Groq API
            - **Model:** llama-3.3-70b-versatile
            - **Interface:** Streamlit

            Each new chat starts its own memory thread, so conversations
            stay isolated from one another.
            """
        )

    if not BACKEND_READY:
        st.markdown('<div class="sidebar-section-label">Setup</div>', unsafe_allow_html=True)
        st.warning("`GROQ_API_KEY` is not set in the environment. Add it to enable chat.", icon="⚠️")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">🤖 LangGraph AI Assistant</h1>
        <div class="app-subtitle">Powered by LangGraph + Groq + Streamlit</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
pending_prompt = st.session_state.pop("pending_prompt", None)

if not st.session_state.message_history:
    st.markdown('<div class="welcome-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card welcome-card">
            <div class="welcome-emoji">👋</div>
            <div class="welcome-title">Welcome to LangGraph AI Assistant</div>
            <div class="welcome-text">
                Ask a question, brainstorm an idea, or paste something you'd like
                help with — responses are generated by Llama 3.3 70B on Groq's
                LPU inference engine, so they land fast.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    suggestions = [
        ("💡", "Explain a concept", "Explain quantum computing like I'm a curious beginner."),
        ("🧑‍💻", "Help me code", "Help me debug a Python function that's throwing a KeyError."),
        ("✍️", "Draft something", "Draft a polite follow-up email after a job interview."),
        ("🧠", "Brainstorm ideas", "Brainstorm 5 unique weekend project ideas for a developer."),
    ]
    cols = st.columns(2)
    for i, (icon, label, full_prompt) in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"{icon}  {label}", key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_prompt = full_prompt
                st.rerun()
else:
    for msg in st.session_state.message_history:
        avatar = "🤖" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            st.markdown(f'<div class="msg-time">{msg["time"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input + generation
# ---------------------------------------------------------------------------
chat_prompt = st.chat_input(
    "Message LangGraph AI Assistant..." if BACKEND_READY else "Set GROQ_API_KEY to start chatting",
    disabled=not BACKEND_READY,
)
prompt = pending_prompt or chat_prompt

if prompt and BACKEND_READY:
    now = datetime.now().strftime("%H:%M")
    st.session_state.message_history.append({"role": "user", "content": prompt, "time": now})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
        st.markdown(f'<div class="msg-time">{now}</div>', unsafe_allow_html=True)

    typing_slot = st.empty()
    typing_slot.markdown(
        """
        <div class="typing-row">
            <div class="typing-avatar">🤖</div>
            <div class="typing-bubble"><span></span><span></span><span></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("", show_time=False):
            response = chatbot.invoke({"messages": [HumanMessage(content=prompt)]}, config=CONFIG)
            ai_content = response["messages"][-1].content
    except Exception as exc:  # keep the UI alive even if the API call fails
        ai_content = f"⚠️ Something went wrong reaching Groq: `{exc}`"

    typing_slot.empty()

    reply_time = datetime.now().strftime("%H:%M")
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(ai_content)
        st.markdown(f'<div class="msg-time">{reply_time}</div>', unsafe_allow_html=True)

    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_content, "time": reply_time}
    )
import streamlit as st
import time
from typing import List, Dict, Any

# ----- YOUR BACKEND (unchanged) -----