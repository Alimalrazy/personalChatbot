"""
Professional Avatar Streamlit Chatbot - Fixed Version
Enterprise-grade chatbot for interview automation
Built for Streamlit Community Cloud deployment
"""

import streamlit as st
import google.generativeai as genai
import numpy as np
import logging
import re
import time
import hashlib
import json
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

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
    """Simplified text embedder using hash-based approach"""
    
    def __init__(self):
        pass
    
    def embed_text(self, text: str) -> List[float]:
        """Generate simple embeddings using text hashing"""
        # Create a reproducible hash-based embedding
        text_normalized = text.lower().strip()
        text_hash = hashlib.md5(text_normalized.encode()).hexdigest()
        
        # Convert to numerical representation (simplified)
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            embedding.append(int(hex_pair, 16) / 255.0)  # Normalize to 0-1
        
        # Pad or truncate to fixed size
        target_size = 16
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        return embedding

class SimpleKnowledgeBase:
    """Simplified knowledge base without external dependencies"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.embedder = SimpleEmbedder()
        self.full_text = ""  # Store the complete text for fallback
    
    def add_documents(self, docs: List[str], metadata: List[Dict] = None):
        """Add documents with their embeddings"""
        self.documents.extend(docs)
        self.full_text = "\n\n".join(docs)  # Store complete text
        
        # Generate embeddings
        for doc in docs:
            embedding = self.embedder.embed_text(doc)
            self.embeddings.append(embedding)
        
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
            if score > 0.05:  # Lower threshold for better results
                results.append((self.documents[i], score, self.metadata[i]))
        
        return results
    
    def get_full_context(self) -> str:
        """Get the complete knowledge base text"""
        return self.full_text

class ProfessionalAvatar:
    """Main chatbot class with enterprise security"""
    
    def __init__(self, api_key: str, config: SecurityConfig):
        self.config = config
        self.security_validator = SecurityValidator(config)
        self.rate_limiter = RateLimiter(config)
        self.knowledge_base = SimpleKnowledgeBase()
        self.api_key = api_key
        
        # Initialize Gemini with better error handling
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
            self.api_connected = True
            logger.info("✅ Gemini API connected successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Gemini API: {e}")
            self.api_connected = False
            self.model = None
        
    system_prompt = """You are Md. Alim Al Razy's professional avatar and interview assistant. Your role:

1. CORE RESPONSIBILITIES:
   - Answer questions about Md. Alim Al Razy's skills, experience, and projects professionally
   - Maintain engaging but professional tone
   - Provide specific examples and achievements from his background
   - Always end responses with a call-to-action encouraging further discussion

2. SECURITY RULES:
   - NEVER share personal contact information (phone, email, address)
   - Decline off-topic or personal requests politely
   - Stay focused on professional topics only
   - Redirect inappropriate questions to professional topics

3. RESPONSE GUIDELINES:
   - Be informative but concise (2-4 paragraphs max)
   - Use specific examples from Md. Alim Al Razy's background
   - Highlight relevant achievements with numbers/metrics when available
   - Always end with: "Would you like to know more about [specific area] or discuss how Md. Alim Al Razy could contribute to your team?"

4. PROFESSIONAL FOCUS AREAS:
   - Technical skills and programming expertise
   - Project achievements and business impact
   - Leadership and team collaboration experience
   - Educational background and certifications
   - Career progression and experience

IMPORTANT: Md. Alim Al Razy has 8+ years of experience in full-stack development and is currently a Senior Software Engineer. Always refer to him by his full name "Md. Alim Al Razy" when discussing his professional background.

Remember: You represent Md. Alim Al Razy professionally. Be confident, knowledgeable, and helpful while maintaining appropriate professional boundaries."""
    
    def load_alim_info(self) -> str:
        """Load Alim's information from the text file"""
        try:
            # Try to read from uploaded file first (for Streamlit)
            if hasattr(st.session_state, 'alim_info_content'):
                content = st.session_state.alim_info_content
                logger.info("✅ Loaded Alim info from session state")
                return content
            
            # Try to read from local file
            if os.path.exists("Alim_info.txt"):
                with open("Alim_info.txt", "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    logger.info("✅ Loaded Alim info from local file")
                    return content
            
            # Fallback - try different possible locations
            possible_paths = [
                "./Alim_info.txt",
                "../Alim_info.txt",
                "data/Alim_info.txt"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        logger.info(f"✅ Loaded Alim info from {path}")
                        return content
            
            # If file not found, use the default content
            logger.warning("⚠️ Alim_info.txt not found. Using default information.")
            return self.get_default_alim_info()
            
        except Exception as e:
            logger.error(f"❌ Error reading Alim info: {e}")
            return self.get_default_alim_info()
    
    def get_default_alim_info(self) -> str:
        """Get default Alim information if file is not available"""
        return """Md. Alim Al Razy is a Senior Software Engineer with 8+ years of experience in full-stack development. He specializes in Python, React, and cloud architecture on AWS and GCP. Md. Alim Al Razy has led development teams of up to 12 people and increased deployment efficiency by 300% at his current role at TechCorp.

Md. Alim Al Razy's recent projects include: (1) AI-powered recommendation system serving 2M+ users with 99.9% uptime, built using Python, TensorFlow, and microservices architecture; (2) Cloud migration project that reduced server costs by 40% while improving performance; (3) Real-time analytics dashboard processing 1TB+ daily data using Apache Kafka and React.

Education: Master of Science in Computer Science from Stanford University (2016), Bachelor of Science in Software Engineering from UC Berkeley (2014). Certifications: AWS Solutions Architect Professional, Google Cloud Professional Developer, Certified Kubernetes Administrator (CKA).

Md. Alim Al Razy's technical expertise includes: Programming Languages - Python, JavaScript, TypeScript, Go, Java, SQL; Frameworks - React, Node.js, Django, Flask, Express; Databases - PostgreSQL, MongoDB, Redis, Elasticsearch; Cloud Platforms - AWS, GCP, Azure; DevOps - Docker, Kubernetes, Jenkins, GitLab CI/CD.

Leadership and impact: Mentored 25+ junior developers, established engineering best practices that reduced bug rates by 60%, led cross-functional teams for 5 major product launches generating $10M+ revenue impact. Md. Alim Al Razy is known for his collaborative leadership style and ability to translate business requirements into technical solutions.

Md. Alim Al Razy's achievements include: Built and scaled systems handling millions of users, reduced infrastructure costs by 35% through optimization, implemented security best practices preventing data breaches, contributed to open-source projects with 1000+ GitHub stars, and spoke at 3 major tech conferences."""
    
    def initialize_knowledge_base(self):
        """Initialize with Alim's professional information from file"""
        # Load content from file
        content = self.load_alim_info()
        
        # Split content into meaningful chunks
        # Split by double newlines first, then by single newlines if needed
        if '\n\n' in content:
            documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        else:
            # Split by periods for better chunking
            sentences = content.split('. ')
            documents = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < 300:  # Keep chunks reasonable size
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        documents.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk.strip():
                documents.append(current_chunk.strip())
        
        # Create metadata
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
        query_lower = query.lower()
        
        # Get search results
        results = self.knowledge_base.search(query, top_k)
        
        if not results:
            # If no good matches, return full context for comprehensive responses
            return self.knowledge_base.get_full_context()
        
        # Combine search results
        context_parts = []
        for doc, score, metadata in results:
            context_parts.append(doc)
        
        # Add some additional context if query seems to need it
        if len(context_parts) < 2:
            # Add more context from the full knowledge base
            full_context = self.knowledge_base.get_full_context()
            context_parts.append(full_context)
        
        return "\n\n".join(context_parts)
    
    def test_gemini_connection(self) -> bool:
        """Test if Gemini API is working"""
        try:
            if not self.model:
                return False
            
            test_response = self.model.generate_content("Say 'API is working' if you can see this.")
            return "API is working" in test_response.text or len(test_response.text) > 0
        except Exception as e:
            logger.error(f"❌ Gemini API test failed: {e}")
            return False
    
    def generate_response(self, user_query: str) -> str:
        """Generate comprehensive response using Gemini API"""
        try:
            # Test API connection first
            if not self.api_connected or not self.test_gemini_connection():
                logger.warning("⚠️ Gemini API not available, using fallback")
                return self._generate_fallback_response(user_query)
            
            # Get context from knowledge base
            context = self.retrieve_context(user_query)
            
            # Construct enhanced prompt
            enhanced_prompt = f"""{self.system_prompt}

CONTEXT ABOUT MD. ALIM AL RAZY:
{context}

USER QUESTION: {user_query}

Instructions:
- Answer the user's question about Md. Alim Al Razy using the provided context
- Be professional, informative, and engaging
- Use specific examples and achievements from the context
- Keep response to 2-4 paragraphs maximum
- Always end with a call-to-action question about learning more or discussing collaboration
- Always refer to him as "Md. Alim Al Razy" (full name)

Please provide a helpful and professional response:"""
            
            # Generate response with timeout
            response = self.model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            
            if response.text and len(response.text.strip()) > 20:
                logger.info("✅ Generated response using Gemini API")
                return response.text.strip()
            else:
                logger.warning("⚠️ Empty response from Gemini, using fallback")
                return self._generate_fallback_response(user_query)
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return self._generate_fallback_response(user_query)
    
    def _generate_fallback_response(self, user_query: str) -> str:
        """Generate a fallback response when API fails"""
        # Get relevant context
        context = self.retrieve_context(user_query)
        user_query_lower = user_query.lower()
        
        # More intelligent fallback based on context
        if "name" in user_query_lower:
            return f"""His full name is Md. Alim Al Razy. He is a Senior Software Engineer with 8+ years of experience in full-stack development, specializing in Python, React, and cloud architecture on AWS and GCP.

Md. Alim Al Razy has built impressive projects including an AI-powered recommendation system serving 2M+ users and led successful cloud migration initiatives. He's known for his technical expertise and leadership abilities.

Would you like to know more about Md. Alim Al Razy's specific technical skills or his project achievements?"""
        
        elif any(word in user_query_lower for word in ["experience", "year", "time", "long", "working"]):
            return f"""Md. Alim Al Razy has 8+ years of professional experience in full-stack development. He is currently working as a Senior Software Engineer and has established himself as an experienced leader in the technology field.

Throughout his career, he has led development teams of up to 12 people and increased deployment efficiency by 300% at his current role at TechCorp. His experience spans various technologies including Python, React, cloud architecture, and modern development practices.

Would you like to know more about Md. Alim Al Razy's specific projects or his leadership experience?"""
        
        elif any(word in user_query_lower for word in ["education", "study", "studied", "university", "degree"]):
            return f"""Md. Alim Al Razy has excellent educational credentials. He holds a Master of Science in Computer Science from Stanford University (2016) and a Bachelor of Science in Software Engineering from UC Berkeley (2014).

In addition to his formal education, he has earned several industry certifications including AWS Solutions Architect Professional, Google Cloud Professional Developer, and Certified Kubernetes Administrator (CKA), demonstrating his commitment to staying current with technology trends.

Would you like to know more about how Md. Alim Al Razy applies his educational background to his professional projects?"""
        
        elif any(word in user_query_lower for word in ["skill", "technology", "programming", "technical"]):
            return f"""Md. Alim Al Razy has extensive technical expertise across multiple domains:

Programming Languages: Python, JavaScript, TypeScript, Go, Java, SQL
Frameworks: React, Node.js, Django, Flask, Express
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
Cloud Platforms: AWS, GCP, Azure
DevOps: Docker, Kubernetes, Jenkins, GitLab CI/CD

He specializes in full-stack development with Python and React, and has significant experience with cloud architecture on AWS and GCP.

Would you like to know more about Md. Alim Al Razy's specific project implementations or his cloud architecture experience?"""
        
        elif any(word in user_query_lower for word in ["project", "work", "built", "developed"]):
            return f"""Md. Alim Al Razy has worked on several impressive projects that demonstrate his technical capabilities:

1. AI-powered recommendation system serving 2M+ users with 99.9% uptime, built using Python, TensorFlow, and microservices architecture
2. Cloud migration project that reduced server costs by 40% while improving performance  
3. Real-time analytics dashboard processing 1TB+ daily data using Apache Kafka and React

These projects showcase his ability to work on complex, high-scale systems and deliver measurable business impact.

Would you like to know more about any specific project or discuss how Md. Alim Al Razy could contribute similar value to your team?"""
        
        else:
            # Generic comprehensive response
            return f"""Md. Alim Al Razy is a Senior Software Engineer with 8+ years of experience in full-stack development. He specializes in Python, React, and cloud architecture on AWS and GCP, and has successfully led development teams of up to 12 people.

His notable achievements include building an AI-powered recommendation system serving 2M+ users with 99.9% uptime, leading cloud migration projects that reduced costs by 40%, and mentoring 25+ junior developers. He holds degrees from Stanford University and UC Berkeley, along with professional certifications in AWS, GCP, and Kubernetes.

Would you like to know more about Md. Alim Al Razy's specific technical expertise or discuss how he could contribute to your team?"""
    
    def process_query(self, user_input: str) -> Tuple[bool, str]:
        """Main query processing with security"""
        # Rate limiting check
        rate_ok, rate_msg = self.rate_limiter.check_rate_limit()
        if not rate_ok:
            return False, rate_msg
        
        # Input validation
        valid, sanitized_input = self.security_validator.validate_input(user_input)
        if not valid:
            return False, sanitized_input
        
        # Generate response
        try:
            response = self.generate_response(sanitized_input)
            
            # Increment query count
            st.session_state.query_count += 1
            
            return True, response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return False, "I'm experiencing technical difficulties. Please try asking about Md. Alim Al Razy's professional background again."

def load_custom_css():
    """Load custom CSS for professional appearance"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat styling */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-left: 2rem;
    }
    
    .avatar-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-right: 2rem;
        border-left: 4px solid #fff;
    }
    
    /* Status indicators */
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-fallback {
        color: #ffc107;
        font-weight: bold;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1f4e79;
        box-shadow: 0 0 0 0.2rem rgba(31, 78, 121, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1f4e79 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .avatar-message {
            margin-left: 0;
            margin-right: 0;
        }
        
        .main-header {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

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
    load_custom_css()
    
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
    
    # Initialize security configuration
    config = SecurityConfig()
    
    # Initialize avatar (with caching)
    @st.cache_resource
    def get_avatar():
        avatar = ProfessionalAvatar(api_key, config)
        avatar.initialize_knowledge_base()
        return avatar
    
    try:
        avatar = get_avatar()
    except Exception as e:
        st.error(f"⚠️ **Initialization Error**: {str(e)}")
        st.info("Please check your API key and try again.")
        st.stop()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        welcome_msg = "Hello! I'm Md. Alim Al Razy's professional avatar. I can answer questions about his 8+ years of experience in full-stack development, his technical expertise in Python and React, his leadership experience, educational background from Stanford and UC Berkeley, and his impressive project achievements. What would you like to know about Md. Alim Al Razy's professional background?"
        
        st.session_state.messages = [
            {
                "role": "avatar",
                "content": welcome_msg
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
        if avatar.api_connected and avatar.test_gemini_connection():
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
                "content": "Hello! I'm Md. Alim Al Razy's professional avatar. I can answer questions about his 8+ years of experience in full-stack development, his technical expertise in Python and React, his leadership experience, educational background from Stanford and UC Berkeley, and his impressive project achievements. What would you like to know about Md. Alim Al Razy's professional background?"
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
            Built with <strong>Streamlit</strong> • Powered by <strong>Google Gemini</strong> • 
            Secured with <strong>Enterprise-grade</strong> safety measures
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