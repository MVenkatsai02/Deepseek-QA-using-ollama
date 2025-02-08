import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling for a modern dark theme
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #121212;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* Text input styling */
    .stTextInput textarea {
        color: #ffffff !important;
        background-color: #2a2a2a !important;
        border-radius: 8px !important;
        border: 1px solid #444 !important;
    }
    
    /* Chat message styling */
    .stChatMessage div {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        padding: 12px !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("DeepSeek using Ollama")
st.caption("🚀 Your personal AI Assistant")

# Sidebar configuration
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. Type your coding or AI-related question in the chat input below.
    2. The AI will process your query and respond accordingly.
    3. Continue the conversation seamlessly.
    
    **Capabilities:**
    - Code Assistance 🧑‍💻
    - Text Generation ✍️
    - Q&A 📚
    
    """)

# Initiate the chat engine
llm_engine = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant and AI text generator. Provide concise, correct solutions and responses "
    "with strategic print statements for debugging. Always respond in English. "
    "You are also an expert in answering the input text professionally."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()