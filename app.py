import streamlit as st
import json
import faiss
import numpy as np
import os
from groq import Groq
import hashlib
import re

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

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vector database
@st.cache_resource
def load_vector_db():
    try:
        # Load the FAISS index
        index = faiss.read_index("index.faiss")
        
        # Load the data
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return index, data
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None, None

# Simple text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fallback embedding method using simple hashing
@st.cache_resource
def create_fallback_embedder(dimension=384):  # Default dimension matching all-MiniLM-L6-v2
    class HashEmbedder:
        def __init__(self, dim):
            self.dim = dim
        
        def encode(self, texts, batch_size=32):
            if isinstance(texts, str):
                texts = [texts]
                
            results = []
            for text in texts:
                # Preprocess the text
                text = preprocess_text(text)
                # Split into words
                words = text.split()
                
                # Create a vector using hash values of words
                vector = np.zeros(self.dim, dtype=np.float32)
                
                for i, word in enumerate(words):
                    # Create a hash of the word
                    hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
                    # Use modulo to get an index within the vector dimension
                    idx = hash_val % self.dim
                    # Set a value at this index
                    vector[idx] += 1.0
                
                # Normalize the vector
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                results.append(vector)
                
            if len(results) == 1:
                return results[0]
            return np.array(results)
    
    return HashEmbedder(dimension)

# Try to load sentence transformer if available
@st.cache_resource
def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Using fallback embedding method due to: {str(e)}")
        # Get the dimension from the FAISS index
        index, _ = load_vector_db()
        if index:
            dimension = index.d
        else:
            dimension = 384  # Default
        return create_fallback_embedder(dimension)

# Initialize Groq client
@st.cache_resource
def load_groq_client():
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            api_key = st.secrets.get("GROQ_API_KEY", None)
        
        if not api_key:
            st.warning("GROQ API key not found. Please set it in the environment or in Streamlit secrets.")
            return None
            
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None

# Search function
def search_similar_content(query, index, data, embedder, top_k=5):
    # Encode the query
    query_vector = embedder.encode(query)
    query_vector = query_vector.astype(np.float32)
    
    # Normalize the vector - important if the index is normalized
    faiss.normalize_L2(query_vector.reshape(1, -1))
    
    # Search the index
    distances, indices = index.search(query_vector.reshape(1, -1), top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(data) and idx >= 0:  # Ensure valid index
            results.append({
                "content": data[idx]["content"],
                "metadata": data[idx].get("metadata", {}),
                "score": float(distances[0][i])
            })
    
    return results

# Function to get response from LLM
def get_llm_response(query, context, chat_history, client):
    # Create the prompt with context
    formatted_context = "\n\n".join([f"Context {i+1}:\n{item['content']}" for i, item in enumerate(context)])
    
    # Create chat history format suitable for the model
    formatted_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_history.append({"role": "user", "content": msg["content"]})
        else:
            formatted_history.append({"role": "assistant", "content": msg["content"]})
    
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

    try:
        # Prepare messages for the chat completion
        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history (limited to last few messages for context)
        if formatted_history:
            messages.extend(formatted_history[-5:])  # Include last 5 messages for context
        
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
        return f"Error getting response from LLM: {str(e)}"

# Option to use full-text search as fallback
def fulltext_search(query, data, top_k=5):
    query = preprocess_text(query)
    query_terms = query.split()
    
    # Score each document based on term frequency
    scored_docs = []
    for i, doc in enumerate(data):
        content = preprocess_text(doc["content"])
        score = 0
        for term in query_terms:
            if term in content:
                score += content.count(term)
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

# Load resources
index, data = load_vector_db()
embedder = load_embedder()
groq_client = load_groq_client()

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input at the bottom (fixed)
st.markdown("---")
prompt = st.chat_input("Ask a question about Tamil astrology...", key="user_input")

# Process the user input
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response (placeholder while processing)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        error_occurred = False
        response = "I'm sorry, I couldn't process your question at this time."
        
        # Try to get a response
        try:
            if index is not None and data is not None:
                if embedder is not None:
                    # Use vector search
                    search_results = search_similar_content(prompt, index, data, embedder, 
                                                           top_k=st.session_state.get("top_k", 5))
                else:
                    # Fallback to full-text search
                    search_results = fulltext_search(prompt, data, 
                                                    top_k=st.session_state.get("top_k", 5))
                
                if groq_client is not None:
                    # Get response from LLM
                    response = get_llm_response(prompt, search_results, st.session_state.messages[:-1], groq_client)
                else:
                    error_occurred = True
                    response = "API client is not available. Please configure your GROQ API key."
            else:
                error_occurred = True
                response = "Vector database couldn't be loaded. Please check your index.faiss and data.json files."
        except Exception as e:
            error_occurred = True
            response = f"An error occurred: {str(e)}"
        
        # Display the final response
        if error_occurred:
            st.error(response)
            message_placeholder.error(response)
        else:
            message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

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
    top_k = st.slider("Number of context passages to use", min_value=1, max_value=10, value=5)
    st.session_state.top_k = top_k
    
    # Search method selection
    if embedder.__class__.__name__ != "SentenceTransformer":
        st.info("Using fallback search method due to offline mode.")
    
    # API Key input (for development/testing)
    st.subheader("API Configuration")
    if not os.environ.get("GROQ_API_KEY") and not st.secrets.get("GROQ_API_KEY", None):
        api_key_input = st.text_input("Enter GROQ API Key:", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("API Key set! Reload the client by refreshing the page.")
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Tamil Astrology Book Assistant v1.0")
