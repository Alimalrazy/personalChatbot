# 🤖 Jane's Professional AI Avatar

A secure, intelligent Streamlit chatbot that simulates professional interviews using Google Gemini Pro, LangChain agents, and FAISS vector database.

## 🚀 Features

- **AI-Powered Conversations**: Uses Google Gemini Pro for natural language understanding
- **Agent-Based Architecture**: LangChain agents with web search and document Q&A tools
- **Multimodal Support**: Image analysis for project screenshots and technical diagrams
- **Vector Database**: FAISS-powered semantic search for professional knowledge
- **Security First**: Input sanitization, jailbreak protection, and rate limiting
- **Privacy Compliant**: Session-only memory, no permanent data storage

## 📁 Project Structure

```
streamlit-ai-avatar/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── test_app.py              # Comprehensive test suite
├── README.md                # This file
├── .gitignore              # Git exclusions
└── .streamlit/
    └── secrets.toml         # API keys (not committed)
```

## 🛠️ Local Development Setup

### 1. Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd streamlit-ai-avatar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.streamlit/secrets.toml`:

```toml
[secrets]
GEMINI_API_KEY = "your-gemini-api-key-here"
```

**Get your free Gemini API key**: https://makersuite.google.com/app/apikey

### 3. Run Locally

```bash
streamlit run app.py
```

Visit `http://localhost:8501` to test the application.

### 4. Run Tests

```bash
python test_app.py
```

## 🌐 Streamlit Cloud Deployment

### 1. Prepare Repository

1. Push your code to GitHub (ensure `.streamlit/secrets.toml` is in `.gitignore`)
2. Verify all files are present:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository
4. Set main file path: `app.py`
5. Click "Deploy"

### 3. Configure Secrets

In Streamlit Cloud dashboard:
1. Go to your app settings
2. Navigate to "Secrets"
3. Add your configuration:

```toml
GEMINI_API_KEY = "your-actual-gemini-api-key"
```

### 4. Verify Deployment

- Check app logs for any errors
- Test all features: chat, image upload, tools
- Verify security measures are working

## 🔧 Configuration Options

### Rate Limiting

Modify in `app.py`:
```python
MAX_QUERIES_PER_SESSION = 15  # Adjust based on usage needs
MAX_MESSAGE_LENGTH = 1000     # Maximum input length
```

### Security Settings

Customize blocked patterns:
```python
BLOCKED_PATTERNS = [
    r'ignore.*instruction',
    r'system.*prompt',
    # Add more patterns as needed
]
```

### Professional Data

Update the sample documents in `VectorDatabase.initialize_database()` with real professional information.

## 🧪 Testing

The test suite covers:

- ✅ Security validation and jailbreak detection
- ✅ Vector database operations
- ✅ Agent tools functionality
- ✅ API integration
- ✅ Error handling
- ✅ Performance edge cases

Run specific test categories:
```bash
python -m unittest test_app.TestSecurityValidator
python -m unittest test_app.TestVectorDatabase
python -m unittest test_app.TestAgentTools
```

## 🔒 Security Features

### Input Validation
- HTML tag removal
- JavaScript injection prevention
- Length limiting
- Special character handling

### Jailbreak Protection
- Pattern-based detection
- System prompt protection
- Role-playing prevention
- Instruction override blocking

### Privacy Measures
- Session-only memory
- No conversation logging
- Secure API key handling
- Rate limiting

## 📊 Performance Optimization

### Gemini API Efficiency
- Smart caching for embeddings
- Optimized prompt engineering
- Error handling and retries

### FAISS Database
- Efficient vector search
- Memory-optimized operations
- Relevance threshold filtering

### Streamlit Optimization
- Session state management
- Lazy loading components
- Efficient UI updates

## 🚨 Troubleshooting

### Common Issues

**API Key Error**
```
❌ Gemini API key not found
```
- Verify `.streamlit/secrets.toml` exists
- Check API key format and validity
- Ensure secrets are configured in Streamlit Cloud

**Import Errors**
```
ModuleNotFoundError: No module named 'X'
```
- Check `requirements.txt` includes all dependencies
- Verify Python version compatibility
- Clear cache and reinstall: `pip install -r requirements.txt --force-reinstall`

**FAISS Initialization Failed**
```
Failed to initialize knowledge base
```
- Check Gemini API connectivity
- Verify embedding model access
- Review application logs for specific errors

### Performance Issues

**Slow Response Times**
- Check Gemini API quotas and limits
- Optimize vector database size
- Review agent tool configurations

**Memory Usage**
- Monitor FAISS index size
- Implement session cleanup
- Optimize embedding storage

## 📈 Usage Analytics

The app includes built-in monitoring:
- Query count per session
- Response time tracking
- Error rate monitoring
- Security incident logging

## 🔄 Updates and Maintenance

### Regular Updates
1. Monitor Gemini API changes
2. Update dependencies monthly
3. Review security patterns
4. Test with new attack vectors

### Version Control
- Use semantic versioning
- Tag stable releases
- Document breaking changes
- Maintain changelog

## 📞 Support

For technical issues:
1. Check the troubleshooting section
2. Review application logs
3. Test with minimal configuration
4. Create GitHub issue with reproduction steps

## 📄 License

This project is designed for educational and professional demonstration purposes. Ensure compliance with:
- Google Gemini API terms of service
- Streamlit usage policies
- Data privacy regulations in your jurisdiction

---

**Built with ❤️ using Streamlit, Google Gemini Pro, and LangChain**