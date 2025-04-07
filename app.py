import streamlit as st
import os
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import re
from groq import Groq
import time

# Set page configuration
st.set_page_config(
    page_title="தமிழ் ஜாதகம் உதவியாளர்",
    page_icon="🔮",
    layout="wide",
)

# Configuration constants - read from environment variables with fallbacks
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", "ocr_vector_db")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f0f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #663399;
    }
    .stTextInput, .stTextArea {
        background-color: white;
    }
    .search-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .result-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .source-text {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #663399;
        font-size: 0.9em;
        margin-top: 10px;
    }
    .tamil-text {
        font-family: 'Noto Sans Tamil', sans-serif;
        line-height: 1.6;
    }
    .info-box {
        background-color: #e6f2ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .error-box {
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        color: #c62828;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Class for vector database with FAISS
class FAISSVectorDB:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        """Initialize FAISS vector database with a multilingual model that supports Tamil."""
        self.model = SentenceTransformer(embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.metadata = []
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text string."""
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.detach().cpu().numpy()
    
    def search(self, query: str, k: int = 5, threshold: float = 100.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks to query with a distance threshold.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Maximum distance threshold for results
            
        Returns:
            List of results with metadata
        """
        if not self.index:
            raise ValueError("Index not created. Call create_index first.")
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Search the index
        distances, indices = self.index.search(
            query_embedding.reshape(1, self.dimension).astype(np.float32), k
        )
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and distances[0][i] < threshold:  # Valid index and within threshold
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][i])
                })
        
        return results
    
    @classmethod
    def load(cls, folder_path: str, model_name: str = EMBEDDING_MODEL):
        """
        Load a vector database from disk.
        
        Args:
            folder_path: Folder containing the saved database
            model_name: Name of the embedding model
            
        Returns:
            Loaded FAISSVectorDB instance
        """
        instance = cls(model_name)
        
        try:
            # Verify folder exists
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Vector database folder not found at: {folder_path}")
            
            data_file = os.path.join(folder_path, "data.json")
            index_file = os.path.join(folder_path, "index.faiss")
            
            # Check if files exist
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"data.json not found in {folder_path}")
                
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"index.faiss not found in {folder_path}")
            
            # Load chunks and metadata
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                instance.chunks = data["chunks"]
                instance.metadata = data["metadata"]
                instance.dimension = data["dimension"]
            
            # Load FAISS index
            instance.index = faiss.read_index(index_file)
            
            print(f"Successfully loaded vector database from {folder_path} with {len(instance.chunks)} chunks")
            return instance
        except Exception as e:
            print(f"Error loading vector database from {folder_path}: {e}")
            raise

# LLM processor using Groq API
class LLMProcessor:
    def __init__(self, api_key: str, model_name: str = LLM_MODEL):
        self.client = Groq(api_key=api_key)
        self.model = model_name
    
    def process_query(self, query: str, context_chunks: List[Dict[str, Any]], 
                      temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Process a query using LLM with retrieved context chunks.
        
        Args:
            query: User query
            context_chunks: List of retrieved context chunks
            temperature: Temperature parameter for LLM
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response
        """
        # Prepare context from retrieved chunks
        context_text = "\n\n".join([chunk["chunk"] for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""You are a Tamil astrology (ஜாதகம்) expert assistant. 
        
USER QUERY: {query}

CONTEXT INFORMATION FROM KNOWLEDGE BASE:
{context_text}

INSTRUCTIONS:
1. Answer the query based ONLY on the context information provided above.
2. If the information to answer the query is not in the context, clearly state "I don't have enough information to answer this question."
3. Do NOT make up or hallucinate any information that is not in the context.
4. Maintain the Tamil astrological terminology in your answer when appropriate.
5. Format your answer clearly with appropriate sections and bullet points if needed.
6. If tables are present in the context, preserve their structure in your response.
7. Your answer should be helpful, accurate, and respectful of Tamil astrological traditions.

YOUR RESPONSE:"""

        try:
            # Make API call to Groq
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing with LLM: {str(e)}"

# Function to format the source text with highlighting
def format_source_text(text: str, query_terms: List[str]) -> str:
    """Format source text with highlighting for query terms."""
    formatted_text = text
    
    # Simple text highlighting
    for term in query_terms:
        if len(term) > 2:  # Only highlight meaningful terms
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            formatted_text = pattern.sub(f"<mark>{term}</mark>", formatted_text)
    
    return formatted_text

# Safe vector DB loading function with multiple fallback paths
def load_vector_db_safely():
    """
    Attempt to load vector DB from various possible locations.
    Returns tuple of (success, vector_db, error_message)
    """
    # List of possible paths to try (in order)
    possible_paths = [
        VECTOR_DB_PATH,  # Try environment variable path first
        os.path.join(os.path.dirname(__file__), VECTOR_DB_PATH),  # Try relative to script
        os.path.join(os.getcwd(), VECTOR_DB_PATH),  # Try relative to current directory
        os.path.abspath(VECTOR_DB_PATH)  # Try absolute path
    ]
    
    # Log paths being tried
    print(f"Attempting to load vector DB from the following paths:")
    for path in possible_paths:
        print(f" - {path}")
    
    # Try each path
    for path in possible_paths:
        try:
            print(f"Attempting to load vector DB from: {path}")
            vector_db = FAISSVectorDB.load(path)
            print(f"Successfully loaded vector DB from: {path}")
            return True, vector_db, None
        except Exception as e:
            print(f"Failed to load from {path}: {str(e)}")
            continue
    
    # If we get here, all paths failed
    return False, None, f"Failed to load vector DB from any path. Check if the database exists and is accessible."

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'vector_db' not in st.session_state or 'db_loaded' not in st.session_state:
        # Attempt to load vector DB
        is_loaded, vector_db, error_msg = load_vector_db_safely()
        
        if is_loaded:
            st.session_state.vector_db = vector_db
            st.session_state.db_loaded = True
        else:
            st.session_state.db_loaded = False
            st.session_state.db_error = error_msg or "Unknown error loading vector database"
    
    if 'llm_processor' not in st.session_state and GROQ_API_KEY:
        try:
            st.session_state.llm_processor = LLMProcessor(GROQ_API_KEY)
            st.session_state.llm_loaded = True
        except Exception as e:
            st.session_state.llm_loaded = False
            st.session_state.llm_error = str(e)
    elif not GROQ_API_KEY:
        st.session_state.llm_loaded = False
        st.session_state.llm_error = "No Groq API key provided"

# Display database status
def show_db_status():
    """Show database loading status."""
    st.sidebar.subheader("System Status")
    
    # Vector DB status
    if st.session_state.get('db_loaded', False):
        st.sidebar.success("✅ Vector database loaded successfully")
        st.sidebar.info(f"Total entries: {len(st.session_state.vector_db.chunks)}")
        st.sidebar.info(f"DB Path: {VECTOR_DB_PATH}")
    else:
        error_msg = st.session_state.get('db_error', 'Unknown error')
        st.sidebar.error(f"❌ Failed to load vector database")
        st.sidebar.error(f"Error: {error_msg}")
        
        # Add a button to retry loading
        if st.sidebar.button("🔄 Retry loading database"):
            # Clear the session state and retry
            if 'vector_db' in st.session_state:
                del st.session_state.vector_db
            if 'db_loaded' in st.session_state:
                del st.session_state.db_loaded
            if 'db_error' in st.session_state:
                del st.session_state.db_error
            st.experimental_rerun()
    
    # LLM status
    if st.session_state.get('llm_loaded', False):
        st.sidebar.success("✅ LLM processor initialized")
        st.sidebar.info(f"Model: {LLM_MODEL}")
    else:
        error_msg = st.session_state.get('llm_error', 'Unknown error')
        st.sidebar.error("❌ LLM processor not initialized")
        st.sidebar.error(f"Error: {error_msg}")

# Main search function
def perform_search(query: str, top_k: int, distance_threshold: float):
    """Perform vector search and LLM processing."""
    if not query.strip():
        return [], None
    
    # Get search results
    start_time = time.time()
    try:
        results = st.session_state.vector_db.search(query, k=top_k, threshold=distance_threshold)
        search_time = time.time() - start_time
        
        # If results found and LLM is available, process with LLM
        llm_answer = None
        llm_time = 0
        if results and st.session_state.get('llm_loaded', False):
            start_time = time.time()
            llm_answer = st.session_state.llm_processor.process_query(query, results)
            llm_time = time.time() - start_time
        
        st.session_state.last_metrics = {
            "search_time": search_time,
            "llm_time": llm_time,
            "results_count": len(results),
        }
        
        return results, llm_answer
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return [], None

# Main application UI
def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("தமிழ் ஜாதகம் உதவியாளர் 🔮")
    st.markdown("<p class='tamil-text'>உங்கள் ஜாதக கேள்விகளுக்கு விடையளிக்க இந்த உதவியாளர் உங்களுக்கு உதவும்</p>", unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Settings")
    show_db_status()
    
    # If database not loaded, show prominent error message
    if not st.session_state.get('db_loaded', False):
        st.error("⚠️ Vector database could not be loaded. Please check the sidebar for details and try the retry button.")
        st.info(f"Looking for vector database at: {VECTOR_DB_PATH}")
        st.info("Make sure the VECTOR_DB_PATH environment variable is set correctly or the database exists in the expected location.")
        
        # Manual path input option
        manual_path = st.text_input("Or specify the vector database path manually:", 
                                  value=VECTOR_DB_PATH)
        if st.button("Try this path"):
            try:
                vector_db = FAISSVectorDB.load(manual_path)
                st.session_state.vector_db = vector_db
                st.session_state.db_loaded = True
                if 'db_error' in st.session_state:
                    del st.session_state.db_error
                st.success(f"Successfully loaded vector database from {manual_path}!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to load from {manual_path}: {str(e)}")
        
        # Exit early if no database
        st.stop()
    
    # Search parameters (only show if DB is loaded)
    top_k = st.sidebar.slider("Number of similar documents to retrieve", 1, 10, 5)
    distance_threshold = st.sidebar.slider("Maximum distance threshold", 10.0, 200.0, 85.0)
    temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.1)
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        show_raw_sources = st.checkbox("Show raw source documents", value=True)
        show_metrics = st.checkbox("Show performance metrics", value=True)
    
    # Search container
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    query = st.text_area("Enter your query about ஜாதகம் (Tamil Astrology)", height=100, 
                        placeholder="Example: ராசி பலன்கள் பற்றி விளக்கவும் or How are doshas determined in Tamil astrology?")
    
    search_col1, search_col2 = st.columns([1, 5])
    with search_col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with search_col2:
        st.markdown("") # Spacing
    st.markdown("</div>", unsafe_allow_html=True)
    
    # If search button is clicked
    if search_button and query:
        with st.spinner("Searching and processing..."):
            results, llm_answer = perform_search(query, top_k, distance_threshold)
        
        # Display results container
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        
        # Display LLM answer if available
        if llm_answer:
            st.markdown("### 📝 Answer")
            st.markdown(f"<div class='tamil-text'>{llm_answer}</div>", unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
        
        # Display metrics if enabled
        if show_metrics and 'last_metrics' in st.session_state:
            metrics = st.session_state.last_metrics
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Search Time", f"{metrics['search_time']:.2f}s")
            metrics_cols[1].metric("LLM Time", f"{metrics['llm_time']:.2f}s")
            metrics_cols[2].metric("Results Count", metrics['results_count'])
        
        # Display source documents if enabled
        if show_raw_sources and results:
            st.markdown("### 📚 Source Documents")
            query_terms = query.split()
            
            for i, result in enumerate(results):
                with st.expander(f"Source {i+1} - Distance: {result['distance']:.2f}"):
                    # Display metadata
                    meta = result['metadata']
                    st.markdown(f"**Contains Tamil**: {'Yes' if meta.get('contains_tamil', False) else 'No'} | "
                                f"**Contains Table**: {'Yes' if meta.get('contains_table', False) else 'No'}")
                    
                    # Format and display source text
                    formatted_text = format_source_text(result['chunk'], query_terms)
                    st.markdown(f"<div class='source-text tamil-text'>{formatted_text}</div>", unsafe_allow_html=True)
        
        elif not results:
            st.warning("No relevant information found in the database. Please try another query.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray;'>தமிழ் ஜாதகம் உதவியாளர் © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
