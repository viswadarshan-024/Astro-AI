import streamlit as st
import json
import faiss
import numpy as np
import os
import hashlib
import re
import time
from typing import List, Dict, Any, Optional, Union, Tuple

# For LLM API calls
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("Groq package not available. Install with: pip install groq")

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Tamil Astrology Book Assistant",
    page_icon="✨",
    layout="wide"
)

# App title and description
st.title("Tamil Astrology Book Assistant")
st.markdown("""
    Ask questions about Tamil astrology and get accurate answers based on the book content.
""")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "embedding_method" not in st.session_state:
    st.session_state.embedding_method = "fallback"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "error_log" not in st.session_state:
    st.session_state.error_log = []

# Utility functions
def log_error(error_message: str) -> None:
    """Add error to session state log with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.error_log.append(f"[{timestamp}] {error_message}")
    if len(st.session_state.error_log) > 10:  # Keep only recent errors
        st.session_state.error_log.pop(0)

def preprocess_text(text: str) -> str:
    """Clean and normalize text for search"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Data loading functions
@st.cache_resource
def load_vector_db() -> Tuple[Optional[Any], Optional[List[Dict[str, Any]]]]:
    """Load FAISS index and data with proper error handling"""
    try:
        # Check if files exist before trying to load them
        if not os.path.exists("index.faiss"):
            log_error("FAISS index file not found")
            return None, None
            
        if not os.path.exists("data.json"):
            log_error("Data JSON file not found")
            return None, None
            
        # Load the FAISS index
        index = faiss.read_index("index.faiss")
        
        # Load the data
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return index, data
    except Exception as e:
        error_msg = f"Error loading vector database: {str(e)}"
        log_error(error_msg)
        return None, None

# Embedding functions
@st.cache_resource
def create_fallback_embedder(dimension: int = 384):
    """Create a fallback embedding method using simple hashing"""
    class HashEmbedder:
        def __init__(self, dim: int):
            self.dim = dim
            self.name = "HashEmbedder"
        
        def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
            """Convert text to vector representations using hash-based approach"""
            if isinstance(texts, str):
                texts = [texts]
                
            results = []
            for text in texts:
                # Preprocess the text
                text = preprocess_text(text)
                # Split into words and n-grams
                tokens = text.split()
                # Add bigrams to improve quality
                bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
                all_tokens = tokens + bigrams
                
                # Create a vector using hash values
                vector = np.zeros(self.dim, dtype=np.float32)
                
                for token in all_tokens:
                    # Create a hash of the token
                    hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
                    # Use modulo to get an index within the vector dimension
                    idx = hash_val % self.dim
                    # Set a value at this index (with TF weighting)
                    vector[idx] += 1.0
                
                # Normalize the vector for cosine similarity
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                results.append(vector)
                
            if len(texts) == 1:
                return results[0]
            return np.array(results)
    
    return HashEmbedder(dimension)

@st.cache_resource
def load_embedder():
    """Load the embedding model with proper fallback"""
    # Try to load sentence transformer if available
    if ST_AVAILABLE:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            st.session_state.embedding_method = "sentence-transformer"
            return model
        except Exception as e:
            log_error(f"Failed to load SentenceTransformer: {str(e)}")
    
    # Fallback to hash-based embeddings
    index, _ = load_vector_db()
    if index:
        dimension = index.d
    else:
        dimension = 384  # Default
        
    st.session_state.embedding_method = "fallback"
    return create_fallback_embedder(dimension)

# LLM API functions
def validate_api_key(api_key: str) -> bool:
    """Validate the API key with a minimal request"""
    if not GROQ_AVAILABLE:
        return False
        
    if not api_key or len(api_key.strip()) < 10:
        return False
        
    try:
        client = Groq(api_key=api_key)
        # Make a minimal API call to validate the key
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return True
    except Exception as e:
        log_error(f"API key validation failed: {str(e)}")
        return False

@st.cache_resource
def load_groq_client():
    """Initialize Groq client with proper error handling"""
    if not GROQ_AVAILABLE:
        log_error("Groq package not installed")
        return None
        
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    # Try to get from Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    
    # Check if we have an API key in session state
    if hasattr(st.session_state, 'groq_api_key') and st.session_state.groq_api_key:
        api_key = st.session_state.groq_api_key
    
    if not api_key:
        return None
    
    try:
        client = Groq(api_key=api_key)
        # Test the client with a minimal query
        if validate_api_key(api_key):
            st.session_state.api_key_configured = True
            return client
        return None
    except Exception as e:
        log_error(f"Error initializing Groq client: {str(e)}")
        return None

# Search functions
def search_similar_content(query: str, index, data: List[Dict], embedder, top_k: int = 5) -> List[Dict]:
    """Search for similar content using vector similarity"""
    try:
        # Handle empty query
        if not query.strip():
            return []
            
        # Encode the query
        query_vector = embedder.encode(query)
        query_vector = query_vector.astype(np.float32)
        
        # Reshape to 2D if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize the vector
        faiss.normalize_L2(query_vector)
        
        # Search the index
        distances, indices = index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(data) and idx >= 0:  # Ensure valid index
                # Convert similarity distance to score (0-1, higher is better)
                similarity_score = 1.0 - min(distances[0][i], 2.0) / 2.0
                
                results.append({
                    "content": data[idx]["content"],
                    "metadata": data[idx].get("metadata", {}),
                    "score": float(similarity_score)
                })
        
        return results
    except Exception as e:
        log_error(f"Search error: {str(e)}")
        return []

def fulltext_search(query: str, data: List[Dict], top_k: int = 5) -> List[Dict]:
    """Basic fallback text search using term frequency"""
    try:
        query = preprocess_text(query)
        query_terms = query.split()
        
        if not query_terms:
            return []
        
        # Score each document based on term frequency
        scored_docs = []
        for i, doc in enumerate(data):
            content = preprocess_text(doc["content"])
            score = 0
            for term in query_terms:
                if term in content:
                    score += content.count(term) / len(content.split())
            if score > 0:
                scored_docs.append({"index": i, "score": score})
        
        # Sort by score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Get top_k documents
        results = []
        for i, doc_info in enumerate(scored_docs[:top_k]):
            idx = doc_info["index"]
            results.append({
                "content": data[idx]["content"],
                "metadata": data[idx].get("metadata", {}),
                "score": doc_info["score"]
            })
        
        return results
    except Exception as e:
        log_error(f"Fulltext search error: {str(e)}")
        return []

# LLM response function
def get_llm_response(query: str, context: List[Dict], chat_history: List[Dict], client) -> str:
    """Get response from LLM with proper error handling"""
    try:
        if not client:
            return "LLM service is not available. Please configure your API key in the sidebar."
            
        # Create the prompt with context
        formatted_context = "\n\n".join([f"Context {i+1}:\n{item['content']}" for i, item in enumerate(context)])
        
        if not formatted_context:
            formatted_context = "No relevant information found in the database."
        
        # Format the system message with instructions
        system_message = """You are an expert Tamil astrology assistant. Answer questions based ONLY on the provided context information.
Important rules:
1. If the context doesn't contain relevant information, say "I don't have enough information about this in the astrology book."
2. Do not hallucinate or make up information not present in the context.
3. Keep answers clear, accurate, and factual based on the Tamil astrology book content.
4. Provide well-structured responses with appropriate headings and formatting when needed.
5. When appropriate, mention that your knowledge comes from the Tamil astrology book.
6. If the question is in Tamil, respond in Tamil; otherwise respond in English.
"""

        # Prepare messages for the chat completion
        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history (limited to last few messages for context)
        formatted_history = []
        for msg in chat_history[-4:]:  # Include last 4 messages for context
            if msg["role"] == "user":
                formatted_history.append({"role": "user", "content": msg["content"]})
            else:
                formatted_history.append({"role": "assistant", "content": msg["content"]})
                
        if formatted_history:
            messages.extend(formatted_history)
        
        # Add the current query with context
        messages.append({
            "role": "user", 
            "content": f"Please answer this question based on the Tamil astrology book information:\n\nQuestion: {query}\n\nRelevant contexts from the book:\n{formatted_context}"
        })
        
        # Get completion from Groq
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            top_p=0.9,
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        log_error(f"LLM error: {error_msg}")
        
        if "Invalid API Key" in error_msg or "invalid_api_key" in error_msg:
            return "Authentication error. Please check your API key in the sidebar settings."
        elif "rate limit" in error_msg.lower():
            return "Rate limit exceeded. Please try again in a few moments."
        else:
            return f"I couldn't generate a response. Error: {error_msg}"

# Load resources
index, data = load_vector_db()
embedder = load_embedder()
groq_client = load_groq_client()

# Sidebar with instructions and customization options
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This application provides insights from a Tamil astrology book using AI technology.
    
    **How to use:**
    1. Type your question in the input box at the bottom
    2. The app will search through the book content
    3. You'll receive an accurate answer based on the book
    """)
    
    st.subheader("Settings")
    
    # Number of search results to use as context
    top_k = st.slider("Context passages", min_value=1, max_value=10, value=st.session_state.top_k)
    st.session_state.top_k = top_k
    
    # API Key configuration
    st.subheader("API Configuration")
    api_key_input = st.text_input("GROQ API Key:", type="password", 
                                 help="Enter your GROQ API key to enable LLM responses")
    
    if api_key_input:
        if validate_api_key(api_key_input):
            st.session_state.groq_api_key = api_key_input
            st.session_state.api_key_configured = True
            st.success("✅ API key valid!")
            # Force refresh of client
            st.cache_resource.clear()
            groq_client = load_groq_client()
        else:
            st.error("❌ Invalid API key. Please check and try again.")
    
    # System status
    st.subheader("System Status")
    
    # Vector DB status
    if index is not None and data is not None:
        st.success(f"✅ Vector DB: {len(data)} passages loaded")
    else:
        st.error("❌ Vector DB: Not loaded")
    
    # Embedding status
    if embedder is not None:
        if ST_AVAILABLE and st.session_state.embedding_method == "sentence-transformer":
            st.success("✅ Embeddings: SentenceTransformer")
        else:
            st.warning("⚠️ Embeddings: Using fallback method")
    else:
        st.error("❌ Embeddings: Not available")
    
    # LLM status
    if st.session_state.api_key_configured:
        st.success("✅ LLM: Connected")
    else:
        st.error("❌ LLM: Not configured")
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Error log (collapsible)
    with st.expander("Error Log", expanded=False):
        if st.session_state.error_log:
            for error in st.session_state.error_log:
                st.text(error)
        else:
            st.text("No errors logged")
    
    st.markdown("---")
    st.caption("Tamil Astrology Book Assistant v1.1")

# Display chat messages
st.markdown("---")
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Function to handle user input
def process_user_input(user_input):
    # Check if we have the necessary components
    if not user_input.strip():
        return
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Default response if everything fails
    default_response = "I'm sorry, I couldn't process your question. Please check the system status in the sidebar."
    response = default_response
    
    try:
        # First check if we have data loaded
        if index is None or data is None:
            response = "The astrology book data couldn't be loaded. Please check if the data files are available."
        else:
            # Try to search for relevant content
            if embedder is not None:
                # Use vector search if possible
                search_results = search_similar_content(user_input, index, data, embedder, top_k=st.session_state.top_k)
            else:
                # Fallback to text search
                search_results = fulltext_search(user_input, data, top_k=st.session_state.top_k)
            
            # If we have API access, get LLM response
            if groq_client is not None and st.session_state.api_key_configured:
                response = get_llm_response(user_input, search_results, st.session_state.messages[:-1], groq_client)
            else:
                # No LLM access, just show search results
                if search_results:
                    response = "API access is not configured. Here are the most relevant passages:\n\n"
                    for i, result in enumerate(search_results):
                        response += f"**Passage {i+1}** (Relevance: {result['score']:.2f}):\n{result['content']}\n\n"
                else:
                    response = "No relevant information found in the astrology book. Please try a different question."
    except Exception as e:
        log_error(f"Processing error: {str(e)}")
        response = f"An error occurred while processing your request: {str(e)}"
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    return response

# Chat input
prompt = st.chat_input("Ask a question about Tamil astrology...", key="user_input")
if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response (placeholder while processing)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Process the input and get response
        response = process_user_input(prompt)
        
        # Update the placeholder with the actual response
        message_placeholder.markdown(response)
    
    # Force a rerun to update the chat history display
    st.rerun()
