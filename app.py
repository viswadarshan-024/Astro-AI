import os
import json
import faiss
import requests
import numpy as np
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import Any, List, Mapping, Optional

# Custom LLM class for Groq API
class GroqLLM(LLM):
    groq_api_key: str
    model_name: str = "llama-3.1-70b-versatile"  # Llama 4 model on Groq
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        
        if stop:
            data["stop"] = stop
            
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from Groq API: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]
    
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

# Setting page configuration
st.set_page_config(
    page_title="Tamil Astrology Assistant",
    page_icon="✨",
    layout="centered"
)

# Custom CSS to make UI clean and neat
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput input {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4b6fff;
        color: white;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Title
st.title("தமிழ் ஜோதிட உதவியாளர் (Tamil Astrology Assistant)")
st.markdown("Ask questions about Tamil astrology and get answers from our knowledge base.")

# Sidebar for API key
with st.sidebar:
    st.title("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key to continue.")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses Tamil astrology texts stored in a vector database.
    It provides answers based on the content of these texts as well as general knowledge.
    Using Llama 4 through Groq API.
    """)

# Function to load the vector store
@st.cache_resource
def load_vector_store():
    try:
        # Load pre-existing FAISS index
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load the vector store from the saved files
        vector_store = FAISS.load_local(".", embeddings, "index")
        
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Function to setup the conversational chain
def setup_conversation_chain(groq_api_key):
    try:
        vector_store = load_vector_store()
        if not vector_store:
            return None
        
        # Set up the Groq LLM with Llama 4
        llm = GroqLLM(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile"  # Llama 4 model on Groq
        )
        
        # Create memory for conversation context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Custom prompt template for Tamil astrology context
        template = """
        You are a knowledgeable Tamil astrology assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the context, just say that you don't know, then use your general knowledge to provide helpful information.
        For greetings and conversational exchanges, just respond naturally without referring to the context.
        Keep track of the conversation history to maintain context across questions.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question:
        {question}
        
        Answer in the same language as the question (Tamil or English):
        """
        
        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        
        # Create the conversational chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return conversation_chain
    
    except Exception as e:
        st.error(f"Error setting up conversation chain: {e}")
        return None

# Display chat history
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Process user input
def process_user_input(user_input):
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Simple greeting detection to avoid unnecessary DB lookups
            greeting_phrases = ["hello", "hi", "vanakkam", "வணக்கம்", "hey", "greet", "good morning", "good afternoon", "good evening"]
            is_greeting = any(greeting in user_input.lower() for greeting in greeting_phrases)
            
            if is_greeting:
                if "தமிழ்" in user_input or any(tamil_word in user_input.lower() for tamil_word in ["வணக்கம்", "நன்றி"]):
                    response = "வணக்கம்! நான் உங்கள் தமிழ் ஜோதிட உதவியாளர். எப்படி உதவ முடியும்?"
                else:
                    response = "Hello! I am your Tamil Astrology Assistant. How can I help you today?"
            else:
                if st.session_state.conversation:
                    try:
                        # Get response from conversation chain
                        result = st.session_state.conversation({"question": user_input})
                        response = result["answer"]
                    except Exception as e:
                        response = f"Sorry, I encountered an error: {str(e)}"
                else:
                    response = "I'm having trouble connecting to the knowledge base. Please make sure the API key is correct."
            
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Main app interface
def main():
    # Display chat history
    display_chat_history()
    
    # Initialize conversation chain if API key is provided
    if groq_api_key and not st.session_state.conversation:
        with st.spinner("Initializing the Tamil Astrology Assistant with Llama 4..."):
            st.session_state.conversation = setup_conversation_chain(groq_api_key)
            if not st.session_state.conversation:
                st.error("Failed to initialize the conversation chain. Please check your API key and try again.")
    
    # Get user input
    user_input = st.chat_input("Ask something about Tamil astrology...")
    
    # Process user input
    if user_input:
        process_user_input(user_input)

if __name__ == "__main__":
    main()
