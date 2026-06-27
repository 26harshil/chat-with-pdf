# Chat With PDF 📄

An interactive, premium Streamlit application that allows users to upload PDF documents and ask questions about them in real-time. Built using LangChain, Hugging Face Embeddings, Pinecone Vector Database, and OpenRouter LLMs.

## Features

- **Document Processing**: Automatically parses and splits uploaded PDF documents into manageable chunks.
- **Semantic Vector Storage**: Indexes chunks into a Serverless Pinecone index (`chatpdf`) using Hugging Face embeddings.
- **RAG Architecture**: Uses retrieval-augmented generation to answer user queries with precise context from the document.
- **Premium UI/UX**: Includes custom styling, responsive layout, sidebar settings, and micro-animations for an elegant user experience.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd chat-with-pdf
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory by copying the `.env.example` template:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and replace the placeholders with your actual API keys:
   - `OPENROUTER_API_KEY` (needed for GPT-4o / GPT-3.5 via OpenRouter)
   - `PINECONE_API_KEY` (needed for vector store indexing)

4. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

## Files in this Repository

- `app.py`: The main Streamlit web application.
- `talk_with_pdf.ipynb`: Jupyter notebook demonstrating the RAG pipeline step-by-step.
- `booking.ipynb`: Experimental notebook for Agentic booking functions.
