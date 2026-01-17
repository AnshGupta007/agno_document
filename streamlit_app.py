#!/usr/bin/env python3
"""
Streamlit Frontend for MindEase - AI Therapist Chatbot
(Modified from Agno Document Q&A System)
Frontend redesigned to provide a calm, conversational therapist-like interface
without changing backend logic.
"""
 
import streamlit as st
import json
import os
from pathlib import Path
import time
from agno_agent import create_agno_document_agent
from local_document_search import initialize_local_search_engine
 
# --------------------------------------------
# 🧠 PAGE CONFIGURATION
# --------------------------------------------
st.set_page_config(
    page_title="MindEase - AI Therapist",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# --------------------------------------------
# 🎨 CUSTOM CSS FOR THERAPY CHAT INTERFACE + GRADIENT BACKGROUND
# --------------------------------------------
st.markdown("""
<style>
    /* Solid black background for the entire app */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
        background: #101010 !important;
    }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
    }

    .stApp {
        font-family: 'Segoe UI', sans-serif;
        color: #e0e0e0;
    }

    .main-header {
        text-align: center;
        color: #81c4ff;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
    }

    .sub-header {
        text-align: center;
        color: #b0b0b0;
        margin-bottom: 2rem;
        font-size: 1rem;
    }

    /* Chat bubble style for dark mode */
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 1rem;
        margin: 0.7rem 0;
        max-width: 75%;
        line-height: 1.6;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.4);
    }

    .user-message {
        background-color: #202d1d;
        color: #dafcd8;
        margin-left: auto;
        border-bottom-right-radius: 0.2rem;
    }

    .assistant-message {
        background-color: #19203a;
        color: #e0e8ff;
        margin-right: auto;
        border-bottom-left-radius: 0.2rem;
    }

    .document-info {
        background-color: #212c1d;
        color: #bee9be;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #57e257;
    }

    .error-message {
        background-color: #3b1a1a;
        color: #ffb2b2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ea3c53;
    }

    .stMetric .stMetricLabel {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

 
# --------------------------------------------
# ⚙️ SESSION STATE INITIALIZATION
# --------------------------------------------
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
 
# --------------------------------------------
# 🧠 LOAD AI AGENT
# --------------------------------------------
def load_agent():
    try:
        if st.session_state.agent is None:
            with st.spinner("🧘 Preparing your AI Therapist..."):
                st.session_state.agent = create_agno_document_agent()
                st.session_state.search_engine = initialize_local_search_engine()
                st.session_state.documents_loaded = len(st.session_state.search_engine.documents) > 0
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        st.error("Make sure GROQ_API_KEY environment variable is set.")
        return False
 
# --------------------------------------------
# 📄 DOCUMENT INFO DISPLAY
# --------------------------------------------
def display_document_info():
    if st.session_state.search_engine and st.session_state.documents_loaded:
        docs = st.session_state.search_engine.documents
        st.markdown('<div class="document-info">', unsafe_allow_html=True)
        st.markdown(f"**📁 Documents Loaded:** {len(docs)}")
 
        with st.expander("View Document Details"):
            for i, doc in enumerate(docs, 1):
                st.write(f"{i}. **{doc['filename']}** ({doc['size']:,} characters)")
 
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-message">', unsafe_allow_html=True)
        st.markdown("⚠️ No documents found. Add files to `agno_document_qa/documents/` folder.")
        st.markdown('</div>', unsafe_allow_html=True)
 
# --------------------------------------------
# 💬 CHAT MESSAGE DISPLAY
# --------------------------------------------
def display_chat_message(role, content):
    if role == "user":
        st.markdown(f'<div class="chat-message user-message">🧑‍💬 <b>You:</b> {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message">🩺 <b>MindEase:</b> {content}</div>', unsafe_allow_html=True)
 
# --------------------------------------------
# 🤔 QUERY PROCESSING
# --------------------------------------------
def process_query(query):
    try:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("MindEase is reflecting..."):
            response = st.session_state.agent.run(query)
            response_text = response.content if hasattr(response, 'content') else str(response)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False
 
# --------------------------------------------
# 🧘 MAIN APPLICATION
# --------------------------------------------
def main():
    initialize_session_state()
 
    # Header
    st.markdown('<h1 class="main-header">🩺 MindEase - Your AI Therapist</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Here to listen, reflect, and guide you through calm conversations.</p>', unsafe_allow_html=True)
 
    # Sidebar
    with st.sidebar:
        st.header("🧩 Configuration")
 
        model_options = [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ]
        st.selectbox("Model:", model_options, index=0)
 
        if st.button("🌿 Initialize MindEase"):
            st.session_state.agent = None
            if load_agent():
                st.success("MindEase is ready to talk 💬")
            else:
                st.error("Initialization failed.")
 
        st.divider()
        st.header("📁 Document Status")
        if st.session_state.search_engine:
            display_document_info()
        else:
            st.info("Initialize MindEase to load documents.")
 
        st.divider()
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
 
        st.divider()
        st.header("💡 Therapy Prompts")
        st.markdown("""
        - How are you feeling today?
        - What has been on your mind lately?
        - What’s causing you stress recently?
        - Can you help me reflect on my emotions?
        """)
 
    # Main Chat Area
    col1, col2 = st.columns([3, 1])
 
    with col1:
        if not st.session_state.agent:
            if not load_agent():
                st.stop()
 
        st.header("💬 Conversation Space")
        for msg in st.session_state.messages:
            display_chat_message(msg["role"], msg["content"])
 
        # New message input
        user_input = st.chat_input("How are you feeling today?")
        if user_input:
            process_query(user_input)
            st.rerun()
 
    with col2:
        st.header("📞 Contact a Therapist")
        st.markdown("""
        If you prefer offline support, consider reaching out to a professional therapist:

        **Dr. Ansh Gupta, PhD** \n
        **Dr. Viraj Kulkarni, PhD**  \n
        Licensed Clinical Psychologist  
        Phone: (123) 456-7890  
        Email: nmimsshirpur@gmail.com  
        Office: NMIMS MPSTME Shirpur

        Office Hours: Mon-Fri, 9am - 5pm  

        Feel free to contact for an appointment or support. \n
                    Keep Smilling :)
        """)
    
if __name__ == "__main__":
    main()