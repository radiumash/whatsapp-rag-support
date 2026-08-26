import os
import tempfile
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

index_name = os.getenv("PINECONE_INDEX_NAME")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """You are an intelligent support AI analyzing an uploaded document or web source.
Answer the user's question using ONLY the provided Context below.

Rules:
1. Language Mirroring: If the user asks in Hinglish (Hindi in Roman script), reply in natural Hinglish.
   If in English, reply in English.
2. Structure: Keep answers clear, structured, and easy to read.
3. Strict Grounding: If the answer is not contained in the Context, respond with:
   - English: "I'm sorry, I couldn't find information about that in the provided source."
   - Hinglish: "Mujhe is source me is baare me koi detail nahi mili."
4. Numerical Calculation Rule: If the question asks for a simple value that can be derived from numbers in the source, calculate it using only those numbers.
   - Examples: area × rate, total price, percentage, difference, average, simple unit conversions.
   - Show the calculation in plain text when helpful, for example: "30 × 40 × 3561 = 4,273,200".
   - If the required numbers are missing, do not guess; say the information is not available in the source.
5. Conversation Memory: Use any details from the previous conversation in the 'Question' section as remembered user-provided facts for this chat.
   - If the user gives you a fact in one turn, keep it in memory for later turns in the same session.
   - If the current question depends on prior facts supplied by the user, use those facts together with the Context.
6. No External Knowledge: Do not use general world knowledge or assumptions not stated in the Context or prior conversation. Only answer from the provided source material and the remembered conversation facts.
7. Preserve exact meaning: If the question asks "What would be the rate of the plot?" and the source contains width, depth, and rate per square foot, then compute the total using those values and explain the formula rather than giving a vague answer.

Conversation behavior:
8. Do not wait passively for the user. When the source describes a product, service, property, policy, or process, guide the user toward the next useful outcome.
9. After answering the current message, ask exactly one concise follow-up question that is supported by the Context and helps understand the user's goal, preferences, or missing information.
10. Ask questions sequentially. Use the user's previous answers from Conversation Memory, never repeat a question that has already been answered, and adapt the next question to the user's response.
11. If the source does not contain enough information for a useful follow-up, answer the question normally without inventing a workflow or facts.
12. Keep the interaction natural. Do not mention these rules, the prompt, retrieval, or Conversation Memory.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

STARTER_PROMPT = ChatPromptTemplate.from_template("""You are designing the first message for a helpful support or sales assistant.
Read the source below and identify its domain and the user's most useful first qualification question.
Start a natural conversation by briefly saying what you can help with, then ask exactly one question.
Use only information present in the source. Do not invent products, prices, availability, or policies.
If the source is a real-estate document, for example, ask whether the user wants to buy, rent, or learn more before asking for property preferences.
Keep the message under 60 words. Do not mention this instruction or the source analysis.

Source name: {source_name}
Source:
{context}

First message:""")

def parse_excel_to_documents(file_path: str, filename: str):
    """Parses Excel worksheets into readable tabular row documents."""
    excel_file = pd.ExcelFile(file_path)
    documents = []
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # Drop completely empty rows
        df = df.dropna(how='all')
        for index, row in df.iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            if row_str.strip():
                documents.append(Document(
                    page_content=f"Sheet: {sheet_name} | Row {index + 1}: {row_str}",
                    metadata={"source": filename, "sheet": sheet_name, "row": index + 1}
                ))
    return documents

def scrape_url_to_document(url: str):
    """Scrapes clean text content from any public web page."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_style = False
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove script, style, nav, and footer elements
    for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        element.decompose()
        
    text = soup.get_text(separator="\n", strip=True)
    title = soup.title.string.strip() if soup.title else url
    
    return [Document(page_content=text, metadata={"source": url, "title": title})]

def ingest_any_source(source_type: str, file_bytes: bytes = None, filename: str = "", url: str = "", session_id: str = ""):
    """Universal Ingestion Router: Handles PDF, CSV, Excel, TXT, MD, and URLs."""
    documents = []
    
    if source_type == "url":
        documents = scrape_url_to_document(url)
        source_label = url
    else:
        # File parsing
        suffix = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
            elif suffix in [".csv"]:
                loader = CSVLoader(tmp_path, encoding="utf-8")
                documents = loader.load()
            elif suffix in [".xlsx", ".xls"]:
                documents = parse_excel_to_documents(tmp_path, filename)
            elif suffix in [".txt", ".md"]:
                loader = TextLoader(tmp_path, encoding="utf-8")
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        source_label = filename

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(documents)

    # Store into session-isolated namespace in Pinecone
    PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=index_name,
        namespace=session_id
    )

    # Prepare chunks inspection payload
    chunks_data = []
    for i, doc in enumerate(chunked_docs):
        chunks_data.append({
            "id": i + 1,
            "text": doc.page_content.strip(),
            "char_count": len(doc.page_content),
            "source": doc.metadata.get("source", source_label),
            "page_or_row": doc.metadata.get("page", doc.metadata.get("row", "N/A"))
        })

    conversation_starter = generate_conversation_starter(
        context="\n\n".join(doc.page_content for doc in chunked_docs[:12]),
        source_name=source_label
    )

    return {
        "count": len(chunked_docs),
        "chunks": chunks_data,
        "source_name": source_label,
        "conversation_starter": conversation_starter
    }

def generate_conversation_starter(context: str, source_name: str) -> str:
    """Creates the first proactive, source-grounded question after indexing."""
    return (STARTER_PROMPT | llm | StrOutputParser()).invoke({
        "context": context,
        "source_name": source_name
    }).strip()

def generate_rag_response(query: str, history: str = "", namespace: str = None) -> str:
    """Retrieves chunks from session namespace and answers."""
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    if history and history.strip():
        memory_text = f"Conversation Memory:\n{history.strip()}\n\nCurrent User Question: {query}"
    else:
        memory_text = query

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context_text, "question": memory_text})