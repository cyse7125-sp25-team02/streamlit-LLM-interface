import streamlit as st
import uuid
import time
from datetime import datetime

# Simulated backend response
def get_bot_response(user_input):
    time.sleep(0.5)
    return f"Bot: Hello! You said: '{user_input}'"

# Custom CSS for improved UI with hidden Streamlit default elements
st.markdown("""
<style>
.stApp {
    background-color: #fafafa;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
    border-right: 1px solid #e0e4e8;
}
.chat-container {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1rem;
    height: 70vh;
    overflow-y: auto;
    border: 1px solid #e0e4e8;
}
.chat-message-user {
    background-color: #4a90e2;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 0.5rem;
}
.chat-message-bot {
    background-color: #f1f3f5;
    color: #333;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    max-width: 70%;
    margin-right: auto;
    margin-bottom: 0.5rem;
}
.timestamp {
    font-size: 0.75rem;
    color: #888;
    margin-top: 0.2rem;
}
.stButton>button {
    background-color: #4a90e2;
    color: white;
    border-radius: 5px;
}
.stTextInput>div>input {
    background-color: #ffffff;
    border-radius: 5px;
    border: 1px solid #d1d5db;
    padding: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stTextInput>div>input:focus {
    border-color: #4a90e2;
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
}
h1, h2, h3 {
    color: #2d3748;
    font-weight: 600;
}

/* Hide Streamlit's default UI elements */
[data-testid="stHeader"] {
    display: none;
}
.stDeployButton {
    display: none;
}
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
header {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = None

# Sidebar for chat management
with st.sidebar:
    st.header("Chatbot")
    if st.button("New Chat"):
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = []
        st.session_state.current_chat = chat_id
    
    for chat_id in st.session_state.chats:
        if st.button(f"Chat {chat_id[:4]}", key=chat_id):
            st.session_state.current_chat = chat_id

# Main chat window
st.header("Chat Window")

if st.session_state.current_chat:
    chat_id = st.session_state.current_chat
    chat_history = st.session_state.chats[chat_id]
    
    # Chat container
    with st.container():
        chat_container = st.container()
        with chat_container:
            for message in chat_history:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message-user">{message["content"]}</div>'
                        f'<div class="timestamp">{datetime.now().strftime("%H:%M")}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message-bot">{message["content"]}</div>'
                        f'<div class="timestamp">{datetime.now().strftime("%H:%M")}</div>',
                        unsafe_allow_html=True
                    )
    
    # User input
    user_input = st.chat_input("Type your message...")
    if user_input:
        chat_history.append({"role": "user", "content": user_input})
        bot_response = get_bot_response(user_input)
        chat_history.append({"role": "assistant", "content": bot_response})
        st.session_state.chats[chat_id] = chat_history
        st.rerun()

else:
    st.write("Create or select a chat to start.")

# Auto-scroll to bottom
st.markdown(
    """
    <script>
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """,
    unsafe_allow_html=True
)
