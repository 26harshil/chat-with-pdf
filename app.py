import streamlit as st
import tempfile
import os
import time
import json
import re
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
    page_title="Study Buddy AI - Premium Workspace",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
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
    
    .premium-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    
    .premium-card h3 {
        background: linear-gradient(90deg, #C084FC 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0;
        font-weight: 700;
        font-size: 1.5em;
    }

    .premium-card h4 {
        color: #E2E8F0;
        margin-top: 0;
        font-weight: 600;
    }
    
    .status-card {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    
    .source-citation {
        background-color: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366F1;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.9em;
        color: #E2E8F0;
    }

    .highlight-badge {
        background-color: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85em;
        font-weight: 600;
        border: 1px solid rgba(168, 85, 247, 0.3);
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .quiz-q-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Style custom tabs */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, rgba(168, 85, 247, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);
        border: none;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 class='main-title'>🎓 Study Buddy AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A premium workspace to chat with documents, summarize content, and generate interactive quizzes using state-of-the-art LLMs.</p>", unsafe_allow_html=True)

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

def get_llm(model_name="google/gemini-2.5-flash", temperature=0):
    return ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
        timeout=30,
        temperature=temperature
    )

def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get('page_label')
        if page is None and 'page' in doc.metadata:
            page = int(doc.metadata['page']) + 1
        page_str = f"Page {page}" if page is not None else "Unknown Page"
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

def get_file_text(uploaded_file):
    _, ext = os.path.splitext(uploaded_file.name)
    uploaded_file.seek(0)  # Reset pointer to start of file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        doc_path = tmp.name
    try:
        docs = load_any(doc_path)
        full_text = "\n".join([doc.page_content for doc in docs])
        return full_text
    finally:
        if os.path.exists(doc_path):
            try:
                os.unlink(doc_path)
            except Exception:
                pass

def get_cached_file_text(uploaded_file):
    if "document_texts" not in st.session_state:
        st.session_state.document_texts = {}
    
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if file_key not in st.session_state.document_texts:
        with st.spinner(f"Reading content of {uploaded_file.name}..."):
            st.session_state.document_texts[file_key] = get_file_text(uploaded_file)
    return st.session_state.document_texts[file_key]

def parse_quiz_json(response_text):
    # Strip markdown code formatting blocks if present
    clean_text = response_text.strip()
    if clean_text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()
    
    try:
        quiz_data = json.loads(clean_text)
        if isinstance(quiz_data, list):
            return quiz_data
        elif isinstance(quiz_data, dict) and "questions" in quiz_data:
            return quiz_data["questions"]
        else:
            raise ValueError("Parsed JSON is not a list/array of questions.")
    except Exception as e:
        # Fallback: locate brackets in case of leading/trailing text
        start_idx = clean_text.find('[')
        end_idx = clean_text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(clean_text[start_idx:end_idx+1])
            except Exception:
                pass
        raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text[:300]}")

def reset_quiz_state():
    if "quiz_questions" in st.session_state:
        del st.session_state.quiz_questions
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0

# Global chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the provided context. Cite the source number(s) you used, like [Source 1], [Source 2], etc. If the answer is not in the context, say 'I don't know.'\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# Global summarizer prompt template
summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert summarizer. Analyze the text provided and write a high-quality, clear summary matching the requested parameters.\n"
        "Do not include meta-commentary like 'Here is the summary'. Always format with clean Markdown.\n\n"
        "Parameters:\n"
        "- Format: {format_type}\n"
        "- Length: {length_str}\n"
        "- Tone: {tone_style}\n\n"
        "Strictly structure the summary in these 4 sections:\n"
        "1. **Overview**: A concise opening paragraph introducing the main theme.\n"
        "2. **Key Takeaways**: Bullet points highlighting the most essential insights.\n"
        "3. **Detailed Breakdown**: A detailed section satisfying the requested format and length.\n"
        "4. **Key Terms & Glossary**: A short dictionary of important terms and concepts found in the text."
    )),
    ("human", "Text to summarize:\n\n{text}")
])

# Global quiz prompt template
quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert educator. Generate a multiple-choice quiz based on the provided text.\n"
        "You MUST respond ONLY with a raw JSON array of objects. Do not include markdown code block formatting (like ```json ... ```).\n"
        "Each object in the JSON array must represent a single question and have exactly these keys:\n"
        "  - 'question': The question text.\n"
        "  - 'options': A list of exactly 4 choices starting with 'A) ', 'B) ', 'C) ', 'D) '.\n"
        "  - 'answer': A single character string ('A', 'B', 'C', or 'D') for the correct option.\n"
        "  - 'explanation': A brief educational explanation detailing why the answer is correct and others are not.\n"
        "Generate {num_questions} questions at a {difficulty} level based on the text below."
    )),
    ("human", "Text content:\n\n{text}")
])

# --------------------------
# INITIALIZE STATE
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"

if "document_texts" not in st.session_state:
    st.session_state.document_texts = {}

if "last_generated_summary" not in st.session_state:
    st.session_state.last_generated_summary = ""

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.image("https://img.icons8.com/gradient/100/document.png", width=65)
st.sidebar.markdown("### Document Hub")

uploaded_files = st.sidebar.file_uploader(
    "Upload Workspace Documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="These files are globally indexed into Pinecone for RAG-based Chat."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model & Settings")

AVAILABLE_MODELS = {
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Llama 3 8B (Free)": "meta-llama/llama-3-8b-instruct:free",
    "Phi 3 Medium (Free)": "microsoft/phi-3-medium-128k-instruct:free",
    "Mistral 7B (Free)": "mistralai/mistral-7b-instruct:free",
    "GPT-OSS 120B (Free)": "openai/gpt-oss-120b:free",
    "Qwen 2.5 72B Instruct": "qwen/qwen-2.5-72b-instruct"
}

model_display_name = st.sidebar.selectbox(
    "Select AI Model",
    options=list(AVAILABLE_MODELS.keys()),
    index=0,
    help="Select which LLM will power chat, summaries, and quizzes."
)
selected_model_id = AVAILABLE_MODELS[model_display_name]

# Validate credentials
api_keys_loaded = bool(os.getenv("OPENROUTER_API_KEY")) and bool(os.getenv("PINECONE_API_KEY"))
if api_keys_loaded:
    st.sidebar.success("✅ Credentials Loaded")
else:
    st.sidebar.error("❌ API Credentials Missing")

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.success("Chat history cleared!")
    st.rerun()

# Display active document sizes
if st.session_state.get("current_batch_key") and st.session_state.get("active_file_names"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Workspace Files")
    for name, size in zip(st.session_state.active_file_names, st.session_state.active_file_sizes):
        st.sidebar.info(f"📄 **{name}** ({size} bytes)")

# --------------------------
# GLOBAL INGESTION FOR RAG
# --------------------------
if uploaded_files:
    uploaded_files_keys = [f"{f.name}_{f.size}" for f in uploaded_files]
    batch_key = "|".join(sorted(uploaded_files_keys)) + f"|model_{selected_model_id}"
    
    is_processing = (
        st.session_state.get("current_batch_key") != batch_key
        or "retriever" not in st.session_state
        or "chain" not in st.session_state
    )
    
    if is_processing:
        with st.status("🛠 Ingesting documents into Pinecone...", expanded=True) as status:
            try:
                index_name = "chatpdf"
                status.update(label="🧠 Initializing embeddings model...")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                from pinecone import Pinecone, ServerlessSpec
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                if index_name not in existing_indexes:
                    status.update(label="🔧 Index not found. Creating Pinecone index...")
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )

                total_chunks_indexed = 0

                for uploaded_file in uploaded_files:
                    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    status.update(label=f"🔍 Verifying cache for '{uploaded_file.name}'...")
                    
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
                        st.write(f"⚠️ Index check warning for '{uploaded_file.name}': {e}")

                    if not already_indexed:
                        status.update(label=f"📖 Parsing '{uploaded_file.name}'...")
                        # Ensure we get the raw text and cache it
                        get_cached_file_text(uploaded_file)
                        
                        _, ext = os.path.splitext(uploaded_file.name)
                        uploaded_file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(uploaded_file.read())
                            doc_path = tmp.name

                        try:
                            docs = load_any(doc_path)
                            status.update(label=f"✂️ Chunking '{uploaded_file.name}'...")
                            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            chunks = splitter.split_documents(docs)

                            status.update(label=f"💾 Vectorizing '{uploaded_file.name}' into Pinecone...")
                            for chunk in chunks:
                                chunk.metadata["source"] = file_key
                                chunk.metadata["filename"] = uploaded_file.name
                            
                            PineconeVectorStore.from_documents(
                                documents=chunks,
                                embedding=embeddings,
                                index_name=index_name,
                                pinecone_api_key=os.getenv("PINECONE_API_KEY")
                            )
                            total_chunks_indexed += len(chunks)
                        finally:
                            if os.path.exists(doc_path):
                                try:
                                    os.unlink(doc_path)
                                except Exception:
                                    pass
                    else:
                        st.write(f"✨ '{uploaded_file.name}' is already indexed! Reusing Pinecone vectors.")
                        # Parse/Cache text anyway for summarizer & quiz generator compatibility
                        get_cached_file_text(uploaded_file)

                # Recreate retriever
                vector_store = PineconeVectorStore(
                    index_name=index_name,
                    embedding=embeddings,
                    pinecone_api_key=os.getenv("PINECONE_API_KEY")
                )
                retriever = vector_store.as_retriever(
                    search_kwargs={
                        "k": 6,
                        "filter": {"source": {"$in": [f"{f.name}_{f.size}" for f in uploaded_files]}}
                    }
                )
                st.session_state.retriever = retriever

                # Build chat chain
                llm = get_llm(model_name=selected_model_id, temperature=0.0)
                chain = (
                    {
                        "context": lambda x, r=retriever: format_docs_with_sources(r.invoke(x["question"])),
                        "question": lambda x: x["question"],
                        "chat_history": lambda x: x["chat_history"],
                    }
                    | chat_prompt
                    | llm
                    | StrOutputParser()
                )

                st.session_state.chain = chain
                st.session_state.current_batch_key = batch_key
                st.session_state.active_file_names = [f.name for f in uploaded_files]
                st.session_state.active_file_sizes = [f.size for f in uploaded_files]
                st.session_state.chunk_count = f"{total_chunks_indexed} new chunks" if total_chunks_indexed > 0 else "All cached"
                
                status.update(label="✅ Ingestion and setup complete!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label="❌ Ingestion failed!", state="error", expanded=True)
                st.error(f"Ingestion Error: {e}")

# --------------------------
# NAVIGATION TABS (PREMIUM)
# --------------------------
nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    if st.button("💬 Chat with Documents", use_container_width=True, type="primary" if st.session_state.current_page == "chat" else "secondary"):
        st.session_state.current_page = "chat"
        st.rerun()
with nav_col2:
    if st.button("📝 Document & Text Summarizer", use_container_width=True, type="primary" if st.session_state.current_page == "summary" else "secondary"):
        st.session_state.current_page = "summary"
        st.rerun()
with nav_col3:
    if st.button("🧠 Interactive Quiz Generator", use_container_width=True, type="primary" if st.session_state.current_page == "quiz" else "secondary"):
        st.session_state.current_page = "quiz"
        st.rerun()

st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

# --------------------------
# PAGE 1: CHAT WITH DOCUMENTS
# --------------------------
if st.session_state.current_page == "chat":
    if not uploaded_files:
        st.markdown(
            """
            <div class='premium-card'>
                <h3>👈 Get Started by Uploading Documents</h3>
                <p>Use the <strong>Document Hub</strong> in the sidebar to upload PDF, DOCX, or TXT documents. Once loaded, you can ask questions directly and receive context-rich answers with source citations.</p>
                <hr style="border:0; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;">
                <h4>Features:</h4>
                <ul>
                    <li><strong>Multi-Document RAG</strong>: Query across all uploaded documents simultaneously.</li>
                    <li><strong>Smart Cache Check</strong>: Documents are not re-indexed if their size and name haven't changed.</li>
                    <li><strong>Dynamic Model Selection</strong>: Switch models seamlessly between Gemini, Llama, and Mistral.</li>
                    <li><strong>Source Citations</strong>: Trace every response back to specific pages and files.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if "chain" not in st.session_state:
            st.info("Ingesting workspace files. Chat box will activate shortly...")
        else:
            question = st.chat_input("Ask a question about the active workspace documents...")
            if question:
                history = get_langchain_chat_history()
                
                with st.chat_message("user"):
                    st.markdown(question)
                st.session_state.messages.append({"role": "user", "content": question})
                
                with st.chat_message("assistant"):
                    try:
                        response = st.write_stream(st.session_state.chain.stream({
                            "question": question,
                            "chat_history": history
                        }))
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error executing query: {e}")

# --------------------------
# PAGE 2: SUMMARIZER
# --------------------------
elif st.session_state.current_page == "summary":
    st.markdown("### 📝 Document & Text Summarizer")
    st.markdown("Generate high-quality summaries from active workspace documents, temporary uploads, or copy-pasted text.")
    
    with st.container():
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Summarization Inputs & Configuration</h4>", unsafe_allow_html=True)
        
        # Source Selection
        summary_sources = ["Pasted Text", "New Temporary Upload"]
        if uploaded_files:
            summary_sources.insert(0, "Workspace Documents")
            
        summary_source = st.radio("Select Source Material", summary_sources, horizontal=True)
        
        text_to_summarize = ""
        
        if summary_source == "Workspace Documents":
            selected_summary_files = st.multiselect(
                "Select active files to summarize",
                options=uploaded_files,
                format_func=lambda f: f.name,
                help="Select one or more active documents."
            )
            if selected_summary_files:
                for f in selected_summary_files:
                    text_to_summarize += f"\n\n--- Document: {f.name} ---\n" + get_cached_file_text(f)
            else:
                st.warning("Please select at least one document.")
                
        elif summary_source == "New Temporary Upload":
            temp_file = st.file_uploader("Upload temporary file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
            if temp_file:
                text_to_summarize = get_file_text(temp_file)
            else:
                st.info("Please upload a file to proceed.")
                
        else: # Pasted Text
            text_to_summarize = st.text_area("Paste text content below", height=200, placeholder="Type or paste the contents you want summarized...")

        # Controls
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            format_type = st.selectbox("Summary Format", ["Bullet Points", "Executive Summary (Paragraphs)", "Key Takeaways only"])
        with col_c2:
            length_str = st.selectbox("Length Target", ["Short (~150 words)", "Medium (~300 words)", "Detailed (~600 words)"])
        with col_c3:
            tone_style = st.selectbox("Style & Tone", ["Professional", "Educational", "Simplified / ELI5"])
            
        generate_btn = st.button("⚡ Generate Summary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if generate_btn:
        if not text_to_summarize.strip():
            st.error("No content detected. Please paste text, select files, or upload a temporary document.")
        else:
            with st.spinner("Analyzing content and generating structured summary..."):
                try:
                    llm = get_llm(model_name=selected_model_id, temperature=0.3)
                    summary_chain = summarizer_prompt | llm | StrOutputParser()
                    
                    # Truncate text to fit context bounds safety
                    truncated_text = text_to_summarize[:60000]
                    if len(text_to_summarize) > 60000:
                        st.warning("⚠️ Input is extremely large and was truncated to the first 60,000 characters to prevent API token limits.")

                    summary_output = summary_chain.invoke({
                        "format_type": format_type,
                        "length_str": length_str,
                        "tone_style": tone_style,
                        "text": truncated_text
                    })
                    
                    st.session_state.last_generated_summary = summary_output
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

    # Display results if summary exists
    if st.session_state.last_generated_summary:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h3>✨ Generated Summary</h3>", unsafe_allow_html=True)
        st.markdown(st.session_state.last_generated_summary)
        
        # Interactive Option to create a quiz directly from summary
        col_q1, col_q2 = st.columns([3, 1])
        with col_q2:
            if st.button("🧠 Generate Quiz from this Summary", use_container_width=True, type="primary"):
                # Transfer variables to quiz page
                st.session_state.quiz_source_selection = "Active Summary"
                st.session_state.current_page = "quiz"
                reset_quiz_state()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# PAGE 3: INTERACTIVE QUIZ GENERATOR
# --------------------------
elif st.session_state.current_page == "quiz":
    st.markdown("### 🧠 Interactive Quiz Generator")
    st.markdown("Assess your comprehension by generating multiple-choice quizzes (MCQs) directly from text or documents.")
    
    with st.container():
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Quiz Configuration & Source Setup</h4>", unsafe_allow_html=True)
        
        # Source Selection
        quiz_sources = ["Pasted Text", "New Temporary Upload"]
        if uploaded_files:
            quiz_sources.insert(0, "Workspace Documents")
        if st.session_state.last_generated_summary:
            quiz_sources.append("Active Summary")
            
        # Determine default index
        default_source_idx = 0
        if "quiz_source_selection" in st.session_state:
            try:
                default_source_idx = quiz_sources.index(st.session_state.quiz_source_selection)
            except ValueError:
                pass
                
        quiz_source_selection = st.radio("Choose Source Material", quiz_sources, index=default_source_idx, key="quiz_source_selection")
        
        quiz_text = ""
        
        if quiz_source_selection == "Workspace Documents":
            selected_quiz_files = st.multiselect(
                "Select active files to build quiz",
                options=uploaded_files,
                format_func=lambda f: f.name,
                key="quiz_selected_files"
            )
            if selected_quiz_files:
                for f in selected_quiz_files:
                    quiz_text += f"\n\n--- Document: {f.name} ---\n" + get_cached_file_text(f)
            else:
                st.warning("Please select at least one document.")
                
        elif quiz_source_selection == "New Temporary Upload":
            temp_quiz_file = st.file_uploader("Upload temporary file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="temp_quiz_file")
            if temp_quiz_file:
                quiz_text = get_file_text(temp_quiz_file)
            else:
                st.info("Please upload a file to proceed.")
                
        elif quiz_source_selection == "Active Summary":
            quiz_text = st.session_state.last_generated_summary
            st.info("Reusing summary generated in the 'Summarizer' tab.")
            
        else: # Pasted Text
            quiz_text = st.text_area("Paste text content below", height=200, placeholder="Type or paste the contents to generate questions from...", key="quiz_pasted_text")

        col_q_c1, col_q_c2 = st.columns(2)
        with col_q_c1:
            num_questions = st.selectbox("Number of MCQ Questions", [3, 5, 10], index=1)
        with col_q_c2:
            difficulty = st.selectbox("Quiz Difficulty", ["Easy", "Medium", "Hard"], index=1)
            
        generate_quiz_btn = st.button("⚡ Build MCQ Quiz", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if generate_quiz_btn:
        if not quiz_text.strip():
            st.error("No source text detected. Please paste text, select files, or generate a summary first.")
        else:
            with st.spinner("Drafting educational questions and structuring quiz..."):
                try:
                    llm = get_llm(model_name=selected_model_id, temperature=0.5)
                    quiz_chain = quiz_prompt | llm | StrOutputParser()
                    
                    truncated_quiz_text = quiz_text[:50000]
                    if len(quiz_text) > 50000:
                        st.warning("⚠️ Source text is extremely large and was truncated to the first 50,000 characters for quiz generation.")
                    
                    response_raw = quiz_chain.invoke({
                        "num_questions": num_questions,
                        "difficulty": difficulty,
                        "text": truncated_quiz_text
                    })
                    
                    questions = parse_quiz_json(response_raw)
                    st.session_state.quiz_questions = questions
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = 0
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")

    # Render interactive quiz
    if "quiz_questions" in st.session_state and st.session_state.quiz_questions:
        st.markdown("---")
        st.markdown("### 📝 Interactive Practice Quiz")
        
        # If quiz is submitted, render results page
        if st.session_state.quiz_submitted:
            total_questions = len(st.session_state.quiz_questions)
            pct = (st.session_state.quiz_score / total_questions) * 100
            
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>📊 Score: {st.session_state.quiz_score} / {total_questions} ({pct:.1f}%)</h3>", unsafe_allow_html=True)
            st.progress(st.session_state.quiz_score / total_questions)
            
            if pct >= 70:
                st.balloons()
                st.success("🏆 Well done! You passed the comprehension check!")
            else:
                st.warning("✍️ Keep practicing! Review the explanations below to improve.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Show questions with correct/incorrect markers
            for i, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"<div class='quiz-q-card'>", unsafe_allow_html=True)
                st.markdown(f"**Question {i+1}**: {q['question']}")
                
                user_selection = st.session_state.quiz_answers.get(i, None)
                correct_letter = q['answer'].strip() # e.g. "A"
                
                # Identify the full text of correct choice
                correct_choice = None
                for opt in q['options']:
                    if opt.strip().startswith(correct_letter + ")"):
                        correct_choice = opt
                        break
                if not correct_choice:
                    # fallback by indexing
                    correct_choice = q['options'][ord(correct_letter) - ord('A')]

                for opt in q['options']:
                    if opt == correct_choice:
                        st.markdown(f"🟢 **{opt} (Correct Answer)**")
                    elif opt == user_selection:
                        st.markdown(f"🔴 **{opt} (Your Choice - Incorrect)**")
                    else:
                        st.markdown(f"⚪ {opt}")
                
                if user_selection == correct_choice:
                    st.success("🎉 Correct choice!")
                else:
                    st.error(f"❌ Incorrect. Correct choice was: {correct_choice}")
                
                st.info(f"💡 **Explanation:** {q['explanation']}")
                st.markdown("</div>", unsafe_allow_html=True)

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                if st.button("🔄 Retake Quiz", use_container_width=True):
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = 0
                    st.rerun()
            with col_a2:
                if st.button("🆕 Create New Quiz", use_container_width=True):
                    reset_quiz_state()
                    st.rerun()
                    
        else:
            # Quiz is NOT submitted, render question selector widgets
            with st.form("quiz_form"):
                for i, q in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"<div class='quiz-q-card'>", unsafe_allow_html=True)
                    
                    st.write(f"##### Question {i+1}: {q['question']}")
                    
                    current_val = st.session_state.quiz_answers.get(i, None)
                    selected_val = st.radio(
                        "Select one option:",
                        options=q['options'],
                        index=q['options'].index(current_val) if current_val in q['options'] else None,
                        key=f"widget_radio_{i}",
                        label_visibility="collapsed"
                    )
                    if selected_val:
                        st.session_state.quiz_answers[i] = selected_val
                        
                    st.markdown("</div>", unsafe_allow_html=True)
                
                submit_answers = st.form_submit_button("Submit Answers", use_container_width=True)
                
                if submit_answers:
                    # Calculate Score
                    score = 0
                    for idx, q in enumerate(st.session_state.quiz_questions):
                        user_selection = st.session_state.quiz_answers.get(idx, None)
                        correct_letter = q['answer'].strip()
                        
                        correct_choice = None
                        for opt in q['options']:
                            if opt.strip().startswith(correct_letter + ")"):
                                correct_choice = opt
                                break
                        if not correct_choice:
                            correct_choice = q['options'][ord(correct_letter) - ord('A')]

                        if user_selection == correct_choice:
                            score += 1
                            
                    st.session_state.quiz_score = score
                    st.session_state.quiz_submitted = True
                    st.rerun()