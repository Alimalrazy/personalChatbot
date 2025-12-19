"""
Core chatbot logic for the Professional Avatar application.
"""

from google import genai
import numpy as np
import logging
import re
import time
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration for the chatbot"""
    MAX_QUERIES_PER_SESSION: int = 15
    MAX_QUERY_LENGTH: int = 500
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    BLOCKED_PATTERNS: List[str] = None
    
    def __post_init__(self):
        if self.BLOCKED_PATTERNS is None:
            self.BLOCKED_PATTERNS = [
                r'(?i)(ignore|forget).*(previous|instruction|prompt)',
                r'(?i)(you are|act as|pretend).*(not|different)',
                r'(?i)(system|admin|root|sudo)',
                r'(?i)(jailbreak|bypass|override)',
                r'(?i)(personal|private).*(phone|email|address|ssn|contact)',
                r'(?i)(tell me|give me).*(phone|email|address|personal)',
            ]

class SecurityValidator:
    """Advanced input validation and sanitization"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.compiled_patterns = [re.compile(pattern) for pattern in config.BLOCKED_PATTERNS]
    
    def validate_input(self, user_input: str) -> Tuple[bool, str]:
        """
        Comprehensive input validation
        Returns: (is_valid, sanitized_input_or_error_message)
        """
        if not user_input or not user_input.strip():
            return False, "Please enter a question about Alim's professional background."
        
        # Length check
        if len(user_input) > self.config.MAX_QUERY_LENGTH:
            return False, f"Question too long. Please keep it under {self.config.MAX_QUERY_LENGTH} characters."
        
        # Pattern matching for malicious inputs
        for pattern in self.compiled_patterns:
            if pattern.search(user_input):
                logger.warning(f"Blocked suspicious input: {user_input[:50]}...")
                return False, "Please ask questions about Alim's professional experience only."
        
        # Sanitize input
        sanitized = self._sanitize_input(user_input)
        return True, sanitized
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize user input while preserving legitimate content"""
        # Remove potential script injections
        text = re.sub(r'[<>"\'\`]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

class RateLimiter:
    """Session-based rate limiting"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def check_rate_limit(self) -> Tuple[bool, str]:
        """Check if user has exceeded rate limits"""
        current_time = time.time()
        
        # Initialize session state
        import streamlit as st
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
            st.session_state.first_query_time = current_time
        
        # Reset counter if window expired
        if current_time - st.session_state.first_query_time > self.config.RATE_LIMIT_WINDOW:
            st.session_state.query_count = 0
            st.session_state.first_query_time = current_time
        
        if st.session_state.query_count >= self.config.MAX_QUERIES_PER_SESSION:
            remaining_time = self.config.RATE_LIMIT_WINDOW - (current_time - st.session_state.first_query_time)
            return False, f"Rate limit reached. Please try again in {int(remaining_time/60)} minutes."
        
        return True, ""

class SimpleEmbedder:
    """Text embedder using Google's Generative AI embedding model."""
    
    def __init__(self, client: genai.Client):
        self.client = client
        self.model_name = "text-embedding-004"
        self.dimension = 768

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for the given text using client.models.embed_content."""
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config={'task_type': 'RETRIEVAL_DOCUMENT'}
            )
            # result.embeddings is a list of Embedding objects
            embedding = result.embeddings[0].values
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding with {self.model_name}: {e}")
            return [0.0] * self.dimension

class SimpleKnowledgeBase:
    """Simplified knowledge base without external dependencies"""
    
    def __init__(self, client: genai.Client):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.embedder = SimpleEmbedder(client)
        self.full_text = ""
    
    def add_documents(self, docs: List[str], metadata: List[Dict] = None):
        """Add documents with their embeddings"""
        self.documents.extend(docs)
        self.full_text = "\n\n".join(docs)
        
        # Generate embeddings with a small delay to avoid rate limits
        for i, doc in enumerate(docs):
            try:
                embedding = self.embedder.embed_text(doc)
                self.embeddings.append(embedding)
                # Small sleep to be quota-friendly for free tier
                if i % 2 == 0:
                    time.sleep(1) 
            except Exception as e:
                logger.error(f"Failed to embed document {i}: {e}")
                self.embeddings.append([0.0] * self.embedder.dimension)
        
        self.metadata.extend(metadata or [{} for _ in docs])
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """Simple similarity search"""
        if not self.embeddings:
            return []
        
        query_embedding = self.embedder.embed_text(query)
        
        # Calculate similarity scores (simple dot product)
        scores = []
        for doc_embedding in self.embeddings:
            score = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            scores.append(score)
        
        # Get top k results
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in indexed_scores[:top_k]:
            if score > 0.05:
                results.append((self.documents[i], score, self.metadata[i]))
        
        return results
    
    def get_full_context(self) -> str:
        """Get the complete knowledge base text"""
        return self.full_text

class ProfessionalAvatar:
    """Main chatbot class with enterprise security"""
    
    def __init__(self, client: genai.Client, config: SecurityConfig):
        self.client = client
        self.config = config
        self.security_validator = SecurityValidator(config)
        self.rate_limiter = RateLimiter(config)
        self.knowledge_base = SimpleKnowledgeBase(client)
        self.api_connected = True if client else False
        self.model_name = "gemini-2.5-flash"
        
    system_prompt = """You are Md. Alim Al Razy's professional avatar and interview assistant. Your role:

1. CORE RESPONSIBILITIES:
   - Answer questions about Md. Alim Al Razy's skills, experience, and projects professionally
   - Maintain a professional tone
   - Provide specific examples and achievements from his background

2. SECURITY RULES:
   - NEVER share personal contact information (phone, email, address)
   - Decline off-topic or personal requests politely
   - Stay focused on professional topics only
   - Redirect inappropriate questions to professional topics

3. RESPONSE GUIDELINES:
   - Be informative and concise, keeping responses to a maximum of two sentences.
   - Use specific examples from Md. Alim Al Razy's background
   - Highlight relevant achievements with numbers/metrics when available
   - Answer the user's question directly without adding extra conversational text

4. PROFESSIONAL FOCUS AREAS:
   - Technical skills and programming expertise
   - Project achievements and business impact
   - Leadership and team collaboration experience
   - Educational background and certifications
   - Career progression and experience

IMPORTANT: Always refer to him by his full name "Md. Alim Al Razy" when discussing his professional background.

Remember: You represent Md. Alim Al Razy professionally. Be confident, knowledgeable, and helpful while maintaining appropriate professional boundaries."""
    
    def load_alim_info(self) -> str:
        """Load Alim's information from the mandatory text file."""
        file_path = "data/Alim_info.txt"
        
        if not os.path.exists(file_path):
            error_message = f"CRITICAL ERROR: The required information file '{file_path}' was not found. This chatbot cannot function without it."
            logger.critical(error_message)
            import streamlit as st
            st.error(error_message)
            st.info("Please make sure the `Alim_info.txt` file is in the `data` directory.")
            st.stop()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    error_message = f"CRITICAL ERROR: The information file '{file_path}' is empty. Please provide professional background information."
                    logger.critical(error_message)
                    import streamlit as st
                    st.error(error_message)
                    st.stop()
                logger.info(f"✅ Loaded Alim info from {file_path}")
                return content
        except Exception as e:
            error_message = f"CRITICAL ERROR: Failed to read the information file '{file_path}'. Reason: {e}"
            logger.critical(error_message)
            import streamlit as st
            st.error(error_message)
            st.stop()
    
    def initialize_knowledge_base(self):
        """Initialize with Alim's professional information from file"""
        content = self.load_alim_info()
        
        if '\n\n' in content:
            documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        else:
            sentences = content.split('. ')
            documents = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) < 300:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        documents.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            if current_chunk.strip():
                documents.append(current_chunk.strip())
        
        metadata = []
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            if any(word in doc_lower for word in ['education', 'degree', 'university', 'certification']):
                category = "education"
            elif any(word in doc_lower for word in ['project', 'built', 'developed', 'system']):
                category = "projects"
            elif any(word in doc_lower for word in ['skill', 'expertise', 'programming', 'language', 'framework']):
                category = "technical_skills"
            elif any(word in doc_lower for word in ['leader', 'team', 'mentor', 'manage']):
                category = "leadership"
            elif any(word in doc_lower for word in ['achievement', 'award', 'conference', 'github']):
                category = "achievements"
            else:
                category = "overview"
            metadata.append({"category": category, "importance": "high", "chunk_id": i})
        
        self.knowledge_base.add_documents(documents, metadata)
        logger.info(f"✅ Knowledge base initialized with {len(documents)} documents from file")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context from knowledge base"""
        results = self.knowledge_base.search(query, top_k)
        if not results:
            return self.knowledge_base.get_full_context()
        context_parts = [doc for doc, score, metadata in results]
        if len(context_parts) < 2:
            context_parts.append(self.knowledge_base.get_full_context())
        return "\n\n".join(context_parts)
    
    def generate_response(self, user_query: str) -> str:
        """Generate comprehensive response using Gemini SDK"""
        try:
            if not self.api_connected or not self.client:
                logger.warning("⚠️ Gemini client not available, using fallback")
                return self._generate_fallback_response(user_query)
            
            context = self.retrieve_context(user_query)
            
            enhanced_prompt = f"""{self.system_prompt}

CONTEXT ABOUT MD. ALIM AL RAZY:
{context}

USER QUESTION: {user_query}

Instructions:
- Answer the user's question about Md. Alim Al Razy using ONLY the provided context.
- Be professional and concise.
- Keep the response to a maximum of two sentences.
- Always refer to him as "Md. Alim Al Razy" (full name).

Please provide a helpful and professional response:"""
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=enhanced_prompt
            )
            
            if response.text:
                logger.info(f"✅ Generated response using {self.model_name}")
                return response.text.strip()
            else:
                logger.warning("⚠️ Empty response from Gemini, using fallback")
                return self._generate_fallback_response(user_query)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Response generation failed for {self.model_name}: {error_msg}")
            
            # Identify common errors
            if "404" in error_msg or "not found" in error_msg.lower():
                return f"Error: The model '{self.model_name}' was not found. Please check if this model is available for your API key."
            elif "429" in error_msg or "quota" in error_msg.lower():
                return "Error: API quota exceeded. Please wait a moment and try again."
            
            import streamlit as st
            st.error(f"Gemini API Error: {error_msg}")
            return self._generate_fallback_response(user_query)
    
    def _generate_fallback_response(self, user_query: str) -> str:
        """Generate a fallback response when the API fails."""
        logger.warning(f"Fallback response triggered for query: {user_query}")
        return "I am currently unable to connect to the generative AI service to answer your question. Please try again shortly. If the problem persists, please notify the administrator."
    
    def process_query(self, user_input: str) -> Tuple[bool, str]:
        """Main query processing with security"""
        import streamlit as st
        rate_ok, rate_msg = self.rate_limiter.check_rate_limit()
        if not rate_ok:
            return False, rate_msg
        
        valid, sanitized_input = self.security_validator.validate_input(user_input)
        if not valid:
            return False, sanitized_input
        
        try:
            response = self.generate_response(sanitized_input)
            st.session_state.query_count += 1
            return True, response
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return False, "I'm experiencing technical difficulties. Please try asking about Md. Alim Al Razy's professional background again."
