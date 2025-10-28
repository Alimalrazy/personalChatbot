import streamlit as st
from pathlib import Path
from src.models.chat import ChatHistory
from src.api.gemini import GeminiClient

class ChatUI:
    def __init__(self):
        self.chat_history = ChatHistory()
        self.gemini_client = GeminiClient()
        self.knowledge_base = self.load_knowledge_base()
        self.system_prompt = self.get_system_prompt()

    def load_knowledge_base(self) -> str:
        try:
            with open("dummy.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            st.error("dummy.txt file not found. Please ensure it exists in the same directory.")
            return ""

    def get_system_prompt(self) -> str:
        return """
You are a helpful assistant. You must answer the user's query *only* using the information provided in the 'Knowledge Base' text.
You are forbidden from using any other information or your general knowledge.

- If the answer to the query can be found in the Knowledge Base, provide the answer.
- If the answer to the query cannot be found in the Knowledge Base, you MUST state that the information is not available in the provided text.
- Do not, under any circumstances, invent information or answer from your general knowledge.
"""

    def prepare_augmented_query(self, prompt: str) -> str:
        return f"""
--- KNOWLEDGE BASE ---
{self.knowledge_base}
----------------------

--- USER QUERY ---
{prompt}
"""

    def run(self):
        st.set_page_config(page_title="Alim Al Razy's Professional Avatar", layout="wide")
        st.title("Alim Al Razy's Professional Avatar")

        # Initialize session state
        if "messages" not in st.session_state:
            self.chat_history.initialize_default()
            st.session_state.messages = self.chat_history.get_messages()

        st.header("Chatbot")
        st.write("Ask anything about his professional life.")

        # Display chat messages
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            if not self.knowledge_base.strip():
                st.error("Please add some text to the Knowledge Base before asking questions.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.write(prompt)

                # Prepare query and get response
                augmented_query = self.prepare_augmented_query(prompt)
                bot_response = self.gemini_client.generate_response(self.system_prompt, augmented_query)

                # Add bot response
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.write(bot_response)
