from typing import List, Dict

class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class ChatHistory:
    def __init__(self):
        self.messages: List[ChatMessage] = []

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role, content))

    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def initialize_default(self):
        if not self.messages:
            self.add_message("assistant", "Hello! Please add some text to the Knowledge Base, and then ask me questions about it.")
