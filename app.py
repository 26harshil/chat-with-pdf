import streamlit as st
import tempfile
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Chat With Docs - Premium Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a premium look
st.markdown("""
<style>
    /* Premium font and styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-weight: 800;
        background: linear-gradient(90deg, #A855F7 0%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        font-size: 2.8em;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.1em;
        margin-bottom: 25px;
    }
    
    .status-card {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    
    .feature-card {
        background-color: #0F172A;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #1E293B;
        margin-top: 15px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
    }

    .feature-card h3 {
        color: #A855F7;
        margin-top: 0;
    }

    .source-citation {
        background-color: #1E293B;
        border-left: 4px solid #A855F7;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 class='main-title'>📄 Chat With Docs</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload PDFs, DOCX, or TXT documents and query them in real time using retrieval-augmented generation with full source citations.</p>", unsafe_allow_html=True)

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def load_any(file_path):
    ext = file_path.lower().split('.')[-1]
    loaders = {
        'pdf': PyPDFLoader,
        'docx': Docx2txtLoader,
        'txt': TextLoader,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    return loaders[ext](file_path).load()

def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
        timeout=30,
        temperature=0
    )

def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        # Parse page number (PyPDFLoader uses page starting from 0)
        page = doc.metadata.get('page_label')
        if page is None and 'page' in doc.metadata:
            page = int(doc.metadata['page']) + 1
        page_str = f"Page {page}" if page is not None else "Unknown Page"
        
        # Format filename
        filename = doc.metadata.get('filename', os.path.basename(doc.metadata.get('source', 'Document')))
        formatted.append(f"[Source {i+1}: {filename} - {page_str}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def get_langchain_chat_history():
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

# Global prompt template supporting history & citations
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the provided context. Cite the source number(s) you used, like [Source 1], [Source 2], etc. If the answer is not in the context, say 'I don't know.'\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# --------------------------
# INITIALIZE STATE
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.image("https://img.icons8.com/gradient/100/document.png", width=70)
st.sidebar.markdown("### Document Hub")

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Support for PDF, DOCX, and TXT documents"
)

# Sidebar controls & info
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings & Controls")

# Verify configuration in sidebar
api_keys_loaded = bool(os.getenv("OPENROUTER_API_KEY")) and bool(os.getenv("PINECONE_API_KEY"))
if api_keys_loaded:
    st.sidebar.success("✅ API Credentials Loaded")
else:
    st.sidebar.error("❌ API Credentials Missing in .env")

# Clear chat history button
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.success("Chat history cleared!")
    st.rerun()

# --------------------------
# CHAT UI & PROCESSING LAYOUT
# --------------------------

# Create containers for UI ordering
chat_container = st.container()
status_container = st.container()

# Display loaded documents information in sidebar
if st.session_state.get("current_batch_key") and st.session_state.get("active_file_names"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Documents Details")
    for name, size in zip(st.session_state.active_file_names, st.session_state.active_file_sizes):
        st.sidebar.info(
            f"📄 **Name:** {name}\n\n"
            f"⚖️ **Size:** {size} bytes"
        )
    st.sidebar.markdown(f"🧩 **Status:** {st.session_state.chunk_count}")

if not uploaded_files:
    # Empty state display in the chat container
    with chat_container:
        st.markdown(
            """
            <div class='feature-card'>
                <h3>👈 Get Started by Uploading Documents</h3>
                <p>Use the sidebar to upload one or multiple PDF, DOCX, or TXT documents. Once uploaded, the assistant will check which documents are already indexed and ingest any new ones into Pinecone.</p>
                <hr style="border: 0; border-top: 1px solid #1E293B; margin: 15px 0;">
                <h4>Features & Upgrades Implemented:</h4>
                <ul>
                    <li><strong>Multi-Document Ingestion</strong>: Upload and query multiple documents at the same time.</li>
                    <li><strong>No Redundant Work</strong>: Checks if each document is already indexed in Pinecone using metadata filtering, preventing duplicate embedding and API charges.</li>
                    <li><strong>Source Citations</strong>: Answers cite precise pages/sections they were generated from, specifying the corresponding document name.</li>
                    <li><strong>Multi-Format Loader</strong>: Native support for PDF, DOCX, and TXT files.</li>
                    <li><strong>Conversational History</strong>: The system retains context for follow-up questions.</li>
                    <li><strong>Robust Error Handling</strong>: Built-in retries and timeouts for LLM API calls.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    # We have uploaded files
    uploaded_files_keys = [f"{f.name}_{f.size}" for f in uploaded_files]
    batch_key = "|".join(sorted(uploaded_files_keys))
    is_processing = st.session_state.get("current_batch_key") != batch_key
    
    # 1. Render Chat History in chat_container
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # 2. Render Chat Input based on processing state
        if is_processing:
            st.chat_input("Processing documents, please wait...", disabled=True)
        else:
            question = st.chat_input("Ask a question about the uploaded documents...")
            if question:
                # Build history first prior to adding the new question to state
                history = get_langchain_chat_history()

                # Display user message
                with st.chat_message("user"):
                    st.markdown(question)
                
                # Save user message to history
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Run retriever and LLM chain with real-time streaming
                with st.chat_message("assistant"):
                    try:
                        # st.write_stream will consume the generator and animate the output
                        response = st.write_stream(st.session_state.chain.stream({
                            "question": question,
                            "chat_history": history
                        }))
                        # Save assistant message to history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun() # rerun to update state/display
                    except Exception as e:
                        st.error(f"Error executing query: {e}")

    # 3. Handle documents processing if needed
    if is_processing:
        with status_container:
            with st.status("🛠 ...", expanded=True) as status:
                try:
                    index_name = "chatpdf"

                    status.update(label="🧠 Loading embeddings model...")
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                    from pinecone import Pinecone, ServerlessSpec
                    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                    
                    # Create index if it does not exist
                    existing_indexes = [idx.name for idx in pc.list_indexes()]
                    if index_name not in existing_indexes:
                        status.update(label="🔧 Index not found. Creating 'chatpdf' index...")
                        pc.create_index(
                            name=index_name,
                            dimension=384,
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )

                    total_chunks_indexed = 0

                    for uploaded_file in uploaded_files:
                        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                        status.update(label=f"🔍 Checking if '{uploaded_file.name}' is already indexed...")
                        
                        already_indexed = False
                        try:
                            index = pc.Index(index_name)
                            results = index.query(
                                vector=[0.0]*384,
                                top_k=1,
                                filter={"source": {"$eq": file_key}},
                                include_metadata=True
                            )
                            already_indexed = len(results.get('matches', [])) > 0
                        except Exception as e:
                            st.write(f"⚠️ Index check failed for '{uploaded_file.name}' (index may be empty or creating): {e}")

                        if not already_indexed:
                            status.update(label=f"📖 Loading '{uploaded_file.name}'...")
                            _, ext = os.path.splitext(uploaded_file.name)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                tmp.write(uploaded_file.read())
                                doc_path = tmp.name

                            try:
                                docs = load_any(doc_path)

                                status.update(label=f"✂️ Splitting '{uploaded_file.name}' into chunks...")
                                splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000,
                                    chunk_overlap=200
                                )
                                chunks = splitter.split_documents(docs)

                                status.update(label=f"💾 Indexing '{uploaded_file.name}' into Pinecone...")
                                # Tag metadata with the individual file_key and the original name
                                for chunk in chunks:
                                    chunk.metadata["source"] = file_key
                                    chunk.metadata["filename"] = uploaded_file.name
                                
                                vector_store = PineconeVectorStore.from_documents(
                                    documents=chunks,
                                    embedding=embeddings,
                                    index_name=index_name,
                                    pinecone_api_key=os.getenv("PINECONE_API_KEY")
                                )
                                total_chunks_indexed += len(chunks)

                            finally:
                                # Clean up temporary file
                                if 'doc_path' in locals() and os.path.exists(doc_path):
                                    try:
                                        os.unlink(doc_path)
                                    except Exception:
                                        pass
                        else:
                            st.write(f"✨ '{uploaded_file.name}' is already indexed! Reusing vectors.")

                    # Re-create references to vector store for retrieval across the batch
                    vector_store = PineconeVectorStore(
                        index_name=index_name,
                        embedding=embeddings,
                        pinecone_api_key=os.getenv("PINECONE_API_KEY")
                    )

                    status.update(label="⛓️ Configuring LangChain chain...")
                    # Set up retriever to filter for any file in the current uploaded batch
                    retriever = vector_store.as_retriever(
                        search_kwargs={
                            "k": 6,  # fetch slightly more chunks since there could be multiple documents
                            "filter": {"source": {"$in": uploaded_files_keys}}
                        }
                    )
                    st.session_state.retriever = retriever

                    # Initialize LLM with retry/error handling
                    llm = get_llm()

                    # Set up chain
                    chain = (
                        {
                            "context": lambda x: format_docs_with_sources(st.session_state.retriever.invoke(x["question"])),
                            "question": lambda x: x["question"],
                            "chat_history": lambda x: x["chat_history"],
                        }
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    # Save objects to session state
                    st.session_state.chain = chain
                    st.session_state.current_batch_key = batch_key
                    st.session_state.active_file_names = [f.name for f in uploaded_files]
                    st.session_state.active_file_sizes = [f.size for f in uploaded_files]
                    st.session_state.chunk_count = f"{total_chunks_indexed} new chunks" if total_chunks_indexed > 0 else "All cached"
                    
                    status.update(label="✅ All documents processed successfully!", state="complete", expanded=False)
                    
                    # Force a rerun to activate the chat input box
                    st.rerun()
                    
                except Exception as e:
                    status.update(label="❌ Processing failed!", state="error", expanded=True)
                    st.error(f"Error: {e}")