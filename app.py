import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Chat With PDF - Premium Assistant",
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
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.2em;
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
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #1E293B;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 class='main-title'>📄 Chat With PDF</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload your documents and query them in real time using retrieval-augmented generation.</p>", unsafe_allow_html=True)

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

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type="pdf",
    help="Support for standard PDF documents"
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

# Display loaded document information in sidebar
if st.session_state.get("current_file_key"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Document Details")
    st.sidebar.info(
        f"📄 **Name:** {st.session_state.pdf_name}\n\n"
        f"🧩 **Chunks:** {st.session_state.chunk_count}"
    )

if not uploaded_file:
    # Empty state display in the chat container
    with chat_container:
        st.markdown(
            """
            <div class='feature-card'>
                <h3>👈 Get Started by Uploading a PDF</h3>
                <p>Use the sidebar to upload a PDF document. Once uploaded, the assistant will split and index the document to answer your questions.</p>
                <hr style="border: 0; border-top: 1px solid #1E293B; margin: 15px 0;">
                <h4>Powered by:</h4>
                <ul>
                    <li><strong>LangChain</strong>: Framework for building LLM applications</li>
                    <li><strong>Pinecone</strong>: Vector database for semantic search</li>
                    <li><strong>HuggingFace Embeddings</strong>: Sentence-transformers for high-quality embedding vectors</li>
                    <li><strong>OpenRouter LLM</strong>: Advanced reasoning to formulate precise answers</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    # We have an uploaded file
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    is_processing = st.session_state.get("current_file_key") != file_key
    
    # 1. Render Chat History in chat_container
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # 2. Render Chat Input based on processing state
        if is_processing:
            st.chat_input("Processing PDF, please wait...", disabled=True)
        else:
            question = st.chat_input("Ask a question about the PDF...")
            if question:
                # Display user message
                with st.chat_message("user"):
                    st.markdown(question)
                
                # Save user message to history
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Run retriever and LLM chain with real-time streaming
                with st.chat_message("assistant"):
                    try:
                        # st.write_stream will consume the generator and animate the output
                        response = st.write_stream(st.session_state.chain.stream(question))
                        # Save assistant message to history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun() # rerun to update state/display
                    except Exception as e:
                        st.error(f"Error executing query: {e}")

    # 3. Handle PDF processing if needed
    if is_processing:
        with status_container:
            with st.status("🛠️ Indexing document chunks in Pinecone...", expanded=True) as status:
                try:
                    # Save to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        pdf_path = tmp.name

                    st.write("📖 Loading PDF pages...")
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()

                    st.write("✂️ Splitting document into chunks...")
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)

                    st.write("🧠 Generating embeddings...")
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                    st.write("💾 Checking and indexing to Pinecone...")
                    from pinecone import Pinecone, ServerlessSpec
                    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                    index_name = "chatpdf"
                    existing_indexes = [idx.name for idx in pc.list_indexes()]
                    if index_name not in existing_indexes:
                        st.write("🔧 Index not found. Creating 'chatpdf' index...")
                        pc.create_index(
                            name=index_name,
                            dimension=384,
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                    
                    vector_store = PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name=index_name,
                        pinecone_api_key=os.getenv("PINECONE_API_KEY")
                    )

                    st.write("⛓️ Configuring LangChain chain...")
                    retriever = vector_store.as_retriever(
                        search_kwargs={"k": 4}
                    )

                    # Initialize LLM using the notebook configuration (OpenRouter / ChatOpenAI)
                    llm = ChatOpenAI(
                        model="openai/gpt-oss-120b:free",
                        api_key=os.getenv("OPENROUTER_API_KEY"),
                        base_url="https://openrouter.ai/api/v1",
                        temperature=0
                    )

                    # Prompt template exactly as specified in the notebook
                    prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"I don't know."
""")

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    chain = (
                        {
                            "context": retriever | format_docs,
                            "question": RunnablePassthrough()
                        }
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    # Save objects to session state
                    st.session_state.chain = chain
                    st.session_state.current_file_key = file_key
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chunk_count = len(chunks)
                    
                    status.update(label="✅ Document processed successfully!", state="complete", expanded=False)
                    
                    # Force a rerun to activate the chat input box
                    st.rerun()
                    
                except Exception as e:
                    status.update(label="❌ Processing failed!", state="error", expanded=True)
                    st.error(f"Error: {e}")
                    
                finally:
                    # Clean up temporary file
                    if 'pdf_path' in locals() and os.path.exists(pdf_path):
                        try:
                            os.unlink(pdf_path)
                        except Exception:
                            pass