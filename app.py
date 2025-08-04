"""
Professional Avatar Streamlit Chatbot - Main Application
"""

import streamlit as st
import google.generativeai as genai
import logging
from chatbot.logic import ProfessionalAvatar, SecurityConfig, SimpleEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_custom_css(file_path: str):
    """Load custom CSS from a file"""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="Md. Alim Al Razy's Professional Avatar",
        page_icon="👨‍💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css("frontend/styles.css")
    
    # Header
    st.markdown("<h1 class='main-header'>👨‍💼 Md. Alim Al Razy's Professional Avatar</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Ask me anything about Md. Alim Al Razy's professional background, skills, and experience!</p>", unsafe_allow_html=True)
    
    # Initialize API key
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key or len(api_key) < 20:
            raise ValueError("Invalid API key")
    except (KeyError, FileNotFoundError, ValueError):
        st.error("⚠️ **Configuration Error**: Gemini API key not found or invalid in secrets.")
        st.info("📋 **How to add secrets on Streamlit Cloud**:")
        st.markdown('''
        1. Go to your app's page on [Streamlit Cloud](https://share.streamlit.io).
        2. Click on the **Settings** button in the top right corner.
        3. Go to the **Secrets** tab.
        4. Add your Gemini API key as a new secret with the following format:
           `GEMINI_API_KEY = "your-api-key-here"`
        ''')
        st.stop()
    
    # Configure Gemini API
    model = None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✅ Gemini API configured successfully.")
    except Exception as e:
        st.error("🔴 **Gemini API Initialization Failed**")
        st.error(f"An error occurred while connecting to the Gemini API: **{str(e)}**")
        st.warning("Please double-check your `GEMINI_API_KEY` in the Streamlit secrets and ensure it is valid.")
        st.info("If the key is correct, the Google Cloud project associated with it may have billing issues or disabled APIs.")
        st.stop()

    # Initialize security configuration
    config = SecurityConfig()
    
    # Initialize avatar
    def get_avatar(genai_module, _model, _config):
        avatar = ProfessionalAvatar(_model, _config)
        avatar.knowledge_base.embedder = SimpleEmbedder(genai_module) # Pass genai_module to embedder
        avatar.initialize_knowledge_base()
        return avatar
    
    try:
        avatar = get_avatar(genai, model, config)
    except Exception as e:
        st.error(f"⚠️ **Chatbot Initialization Error**: {str(e)}")
        st.info("There was an issue setting up the chatbot. Please refresh the page to try again.")
        st.stop()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "avatar",
                "content": "Hello! I am Md. Alim Al Razy's professional avatar. How can I help you today?"
            }
        ]
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 👋 Welcome!")
        st.markdown("""
        I'm Md. Alim Al Razy's AI-powered professional representative. Feel free to ask about:
        
        **🎯 Professional Topics:**
        - Technical skills & expertise
        - Project experience & achievements  
        - Educational background
        - Leadership & team experience
        - Career highlights & impact
        """)
        
        # API Status
        st.markdown("### 🔗 System Status")
        if avatar.api_connected:
            st.markdown('<p class="status-connected">🟢 Gemini API Connected</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-fallback">🟡 Using Fallback Mode</p>', unsafe_allow_html=True)
        
        st.markdown("### 🔒 Privacy & Security")
        st.markdown(f"""
        - **Session-only**: No data stored permanently
        - **Rate limited**: {config.MAX_QUERIES_PER_SESSION} questions per hour
        - **Secure**: Input validation & sanitization
        - **Professional**: Work-focused discussions only
        """)
        
        # Query counter
        if 'query_count' in st.session_state:
            remaining = max(0, config.MAX_QUERIES_PER_SESSION - st.session_state.query_count)
            st.markdown(f"**Questions remaining:** {remaining}")
            
            if remaining <= 3:
                st.warning("⏰ You're approaching the rate limit!")
        
        st.markdown("### 📞 Next Steps")
        st.markdown("""
        Impressed by Md. Alim Al Razy's background? 
        
        **Ready to connect?** Ask me about scheduling an interview or getting in touch!
        """)
    
    # Main chat interface
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class='user-message'>
                <strong>You:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='avatar-message'>
                <strong>🤖 Md. Alim Al Razy's Avatar:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # User input section
    col1, col2 = st.columns([4, 1])
    
    # Use a unique key that changes after successful submission
    input_key = f"user_input_{st.session_state.get('input_counter', 0)}"
    
    with col1:
        user_input = st.text_input(
            "Ask your question:",
            placeholder="e.g., What are Md. Alim Al Razy's main technical skills?",
            key=input_key,
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    # Process user input
    if ask_button and user_input:
        # Check if this is a new message (not already in history)
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process query with loading indicator
            with st.spinner("🤔 Thinking..."):
                success, response = avatar.process_query(user_input)
            
            if success:
                st.session_state.messages.append({"role": "avatar", "content": response})
                st.success("✅ Response generated!")
                # Increment input counter to create new input field
                st.session_state.input_counter = st.session_state.get('input_counter', 0) + 1
            else:
                st.error(f"⚠️ {response}")
            
            # Rerun to update the UI
            st.rerun()
    
    # Alternative: Add a "Clear Chat" button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {
                "role": "avatar",
                "content": "Hello! I am Md. Alim Al Razy's professional avatar. How can I help you today?"
            }
        ]
        st.session_state.query_count = 0
        st.session_state.input_counter = st.session_state.get('input_counter', 0) + 1
        st.rerun()
    
    # Debug information (only show in development)
    if st.sidebar.button("🔧 Debug Info"):
        with st.expander("Debug Information"):
            st.write("**API Status:**", "Connected" if avatar.api_connected else "Disconnected")
            st.write("**Knowledge Base Docs:**", len(avatar.knowledge_base.documents))
            st.write("**Session Queries:**", st.session_state.get('query_count', 0))
            
            if avatar.knowledge_base.documents:
                st.write("**First Document Preview:**")
                st.text(avatar.knowledge_base.documents[0][:200] + "...")
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <h3>🚀 About This Avatar</h3>
        <p>
            
        </p>
        <p>
            <small>
                This AI avatar represents Md. Alim Al Razy professionally. All responses are generated 
                based on his professional background and achievements.
            </small>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
