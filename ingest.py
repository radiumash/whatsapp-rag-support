import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Check for required API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY in .env file.")
if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_INDEX_NAME"):
    raise ValueError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env file.")

def run_ingestion():
    docs_dir = "./docs"
    
    if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
        print(f"Error: Directory '{docs_dir}' is empty or does not exist. Add your PDF files there.")
        sys.exit(1)

    print("Step 1/4: Loading PDF documents from ./docs...")
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"-> Successfully loaded {len(documents)} document pages.")

    print("\nStep 2/4: Chunking text for RAG...")
    # 500 characters chunk size with 50 character overlap keeps related facts together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"-> Created {len(chunked_docs)} text chunks.")

    print("\nStep 3/4: Initializing OpenAI Multilingual Embeddings (3072 dimensions)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    index_name = os.getenv("PINECONE_INDEX_NAME")
    print(f"\nStep 4/4: Uploading vector embeddings to Pinecone index '{index_name}'...")
    
    # Store documents into Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=index_name
    )
    
    print("\n🎉 Document Ingestion Complete! Your knowledge base is now indexed.")

if __name__ == "__main__":
    run_ingestion()