import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from groq import Groq

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

# Initialize the sentence transformer model
@st.cache_resource
def load_sentence_transformer():
    try:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    except Exception as e:
        st.error(f"Error initializing sentence transformer: {str(e)}")
        return None

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
def search_similar_content(query, index, data, model, top_k=5):
    # Encode the query
    query_vector = model.encode([query])[0]
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
        
        # Add chat history
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
            max_tokens=6000,
            top_p=0.9,
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        return f"Error getting response from LLM: {str(e)}"

# Load resources
index, data = load_vector_db()
sentence_transformer = load_sentence_transformer()
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
if prompt and index is not None and data is not None and sentence_transformer is not None and groq_client is not None:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response (placeholder while processing)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Search for relevant content
        search_results = search_similar_content(prompt, index, data, sentence_transformer)
        
        # Get response from LLM
        response = get_llm_response(prompt, search_results, st.session_state.messages[:-1], groq_client)
        
        # Display the final response
        message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display a warning if any component failed to load
if index is None or data is None:
    st.error("Failed to load the vector database. Please check your index.faiss and data.json files.")
if sentence_transformer is None:
    st.error("Failed to load the sentence transformer model.")
if groq_client is None:
    st.error("Failed to initialize Groq client. Please set your GROQ_API_KEY.")

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
