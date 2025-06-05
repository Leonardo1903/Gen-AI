from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()


# openai_api_key=os.getenv("OPENAI_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Load Pdf
pdf_path= Path(__file__).parent / "nodejs.pdf"

loader= PyPDFLoader(file_path=pdf_path)
docs= loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

split_docs = text_splitter.split_documents(docs)

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=openai_api_key
# )

# Generate vector embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

# Addition of embeddings to Vector DB
# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="nodejs",
#     embedding=embeddings,
# )

# vector_store.add_documents(split_docs)
# print ("Documents added to Qdrant vector store.")

user_question = input("Ask a question about Node.js: ")

# Retrieval of relevant chunks from Vector DB
retriever=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="nodejs",
    embedding=embeddings,
)


# relevant_chunks=search_result= retriever.similarity_search(
#     query="what is the FS module in Node.js?",
# )

# print ("Relevant chunks:",search_result)

search_results = retriever.similarity_search(
    query=user_question
)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

    Context:
    {context}
"""

chat_completion = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": user_question },
    ]
)

print(f"🤖: {chat_completion.choices[0].message.content}")

