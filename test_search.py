import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
index_name = os.getenv("PINECONE_INDEX_NAME")

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

def test_query(query: str):
    print(f"\nQuery: '{query}'")
    print("-" * 50)
    # Search for top 2 matching chunks
    results = vectorstore.similarity_search_with_score(query, k=2)
    
    for doc, score in results:
        print(f"Score: {score:.4f} | Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content.strip()}")
        print("-" * 30)

if __name__ == "__main__":
    # Test English query
    test_query("How many days do I have to return an item?")
    
    # Test Hinglish query
    test_query("Order cancel hone par refund kitne din me aayega?")