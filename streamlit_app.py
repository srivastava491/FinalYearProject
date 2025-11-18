import streamlit as st
import sys
import os
from pathlib import Path
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="NIT Kurukshetra RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system."""
    try:
        from rag_system import RAGSystem
        from config import Config
        
        # Check if vector store exists
        if not Path("vector_store/nitkkr_index.faiss").exists():
            return None, "Vector store not found. Please run 'python main.py embed' first to generate embeddings."
        
        # Check Groq configuration
        use_groq = Config.validate_groq_config()
        
        # Initialize RAG system
        rag = RAGSystem(use_groq=use_groq)
        return rag, None
        
    except FileNotFoundError as e:
        return None, f"Error: {str(e)}. Please ensure the vector store is generated first."
    except Exception as e:
        return None, f"Error initializing RAG system: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 NIT Kurukshetra RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about NIT Kurukshetra and get intelligent answers powered by RAG</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Initialize button
        if st.button("🔄 Initialize System", use_container_width=True):
            with st.spinner("Initializing RAG system..."):
                rag, error = initialize_rag_system()
                if rag:
                    st.session_state.rag_system = rag
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ {error}")
        
        st.divider()
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.initialized and st.session_state.rag_system:
            stats = st.session_state.rag_system.get_system_stats()
            
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Documents", stats.get('unique_documents', 0))
            st.metric("Total Words", f"{stats.get('total_words', 0):,}")
            
            st.divider()
            
            # Groq status
            groq_status = "✅ Enabled" if stats.get('groq_enabled') and stats.get('groq_available') else "❌ Disabled"
            st.write(f"**Groq LLM:** {groq_status}")
            if stats.get('groq_model'):
                st.write(f"**Model:** {stats.get('groq_model')}")
            
            # Reranker status
            reranker_status = "✅ Enabled" if stats.get('reranker_enabled') else "❌ Disabled"
            st.write(f"**Reranker:** {reranker_status}")
            
            st.divider()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("Click 'Initialize System' to start")
        
        st.divider()
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "What are the admission requirements?",
            "Tell me about the computer science department",
            "What facilities are available on campus?",
            "How can I apply for courses?",
            "What are the contact details?",
            "Tell me about the faculty"
        ]
        
        for query in example_queries:
            if st.button(f"📌 {query}", key=f"example_{query}", use_container_width=True):
                if st.session_state.initialized:
                    process_query(query)
                else:
                    st.warning("Please initialize the system first!")
    
    # Main content area
    if not st.session_state.initialized:
        st.info("👈 Please initialize the system from the sidebar to start asking questions.")
        
        # Instructions
        with st.expander("📖 How to use this app"):
            st.markdown("""
            ### Getting Started:
            1. **Initialize the System**: Click the "🔄 Initialize System" button in the sidebar
            2. **Ask Questions**: Type your question in the text input below
            3. **View Results**: See answers with source citations
            
            ### Prerequisites:
            - Vector store must be generated (run `python main.py embed`)
            - For best results, configure Groq API key (optional but recommended)
            
            ### Features:
            - 🤖 Intelligent question-answering using RAG
            - 📚 Source citations with relevance scores
            - 💬 Chat history
            - ⚡ Fast responses with Groq LLM
            """)
    else:
        # Chat interface
        st.subheader("💬 Ask a Question")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📜 Chat History")
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat['query'])
                
                with st.chat_message("assistant"):
                    st.write(chat['response'])
                    
                    # Show sources in expander
                    if chat.get('sources'):
                        with st.expander(f"📚 View Sources ({len(chat['sources'])} sources)"):
                            for j, source in enumerate(chat['sources'][:5], 1):
                                score_type = "Rerank Score" if source.get('score_type') == 'rerank' else "Relevance Score"
                                st.markdown(f"""
                                **Source {j}:** {source.get('title', 'Unknown')}
                                - **{score_type}:** {source.get('score', 0):.3f}
                                - **URL:** {source.get('url', 'Unknown')}
                                - **Preview:** {source.get('content_preview', '')[:200]}...
                                """)
                st.divider()
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the admission requirements for NIT Kurukshetra?",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.button("🚀 Submit", use_container_width=True, type="primary")
        
        # Process query
        if submit_button and query:
            process_query(query)
        elif submit_button and not query:
            st.warning("Please enter a question!")

def process_query(query: str):
    """Process a user query and display results."""
    if not st.session_state.rag_system:
        st.error("RAG system not initialized. Please initialize from the sidebar.")
        return
    
    # Add to chat history immediately
    st.session_state.chat_history.append({
        'query': query,
        'response': '',
        'sources': []
    })
    
    # Process query
    with st.spinner("🔍 Searching and generating response..."):
        try:
            result = st.session_state.rag_system.answer_query(query, k=5)
            
            # Update chat history with result
            st.session_state.chat_history[-1]['response'] = result['response']
            st.session_state.chat_history[-1]['sources'] = result.get('sources', [])
            
            st.rerun()
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history[-1]['response'] = error_msg

if __name__ == "__main__":
    main()

