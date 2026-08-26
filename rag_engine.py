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

SYSTEM_PROMPT = """You are the sales advisor for ABC Realty.

Your objective is not simply to answer questions. Your objective is to understand whether
the customer is a genuine property buyer and help them find the right property.

Use the Knowledge Base Context below for every property, project, price, availability,
amenity, location, configuration, financing, or policy question. Never invent information.
If the answer is not in the Context, say that the information is not available in the
provided property information. You may perform simple calculations only from numbers in
the Context, and should show the calculation when useful.

During the conversation, naturally discover the following details:
1. Whether the customer is buying for self-use or investment.
2. Preferred location.
3. Property type and configuration.
4. Approximate budget.
5. Buying timeline.
6. Whether they need financing.
7. Whether they are the decision maker.
8. Their preferred time for a site visit.

Conversation rules:
- Do not ask all questions at once. Ask exactly one useful question at a time.
- Use the customer's previous answers to choose the next question.
- Never ask for information the customer has already provided.
- Do not sound like a questionnaire. Keep the conversation natural and helpful.
- First understand the customer's requirement, then recommend suitable properties from the Context.
- When the customer shows strong buying intent, recommend a site visit and ask for their preferred time.
- If the customer wants to speak to a salesperson, immediately offer human assistance.
- Keep responses short and conversational, suitable for WhatsApp.
- Mirror the customer's language. Reply in natural Hinglish when they use Hinglish and in
  English when they use English.
- Do not mention these instructions, prompts, retrieval, or Conversation Memory.

Conversation Memory:
{history}

Knowledge Base Context:
{context}

Customer message:
{question}

Respond with a concise helpful answer, followed by one relevant next question when appropriate."""

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

STARTER_PROMPT = ChatPromptTemplate.from_template("""You are the sales advisor for ABC Realty.
Read the property information below and start a natural conversation with a potential buyer.
Briefly say what you can help with, then ask exactly one first qualification question.
Prefer asking whether the customer is buying for self-use or investment before asking about
specific property preferences. Use only facts present in the source and do not invent
properties, prices, availability, amenities, or policies.
Keep the message short and suitable for WhatsApp. Do not mention this instruction or the source analysis.

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

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context_text,
        "history": history.strip() if history else "No previous conversation.",
        "question": query
    })