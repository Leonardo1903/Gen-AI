from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# openai_api_key=os.getenv("OPENAI_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")

pdf_path= Path(__file__).parent / "nodejs.pdf"

loader= PyPDFLoader(file_path=pdf_path)
docs= loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

split_docs = text_splitter.split_documents(docs)

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=openai_api_key
# )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

# Ingestion
# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="nodejs",
#     embedding=embeddings,
# )

# vector_store.add_documents(split_docs)
# print ("Documents added to Qdrant vector store.")

user_question = input("Ask a question about Node.js: ")

# Retrieval
retriever=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="nodejs",
    embedding=embeddings,
)


# relevant_chunks=search_result= retriever.similarity_search(
#     query="what is the FS module in Node.js?",
# )

# print ("Relevant chunks:",search_result)

relevant_chunks = retriever.similarity_search(
    query=user_question,
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that answers questions based on the provided context.

    Context:
    {context}

    Question:
    {question}
    """
)

# Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key=gemini_api_key
)

# Format the context for the agent
context_text = "\n".join([doc.page_content for doc in docs])

# Prepare the prompt
formatted_prompt = prompt.format(
    context=context_text,
    question=user_question
)

# Get the answer from Gemini
response = llm.invoke(formatted_prompt)

print("Agent answer:", response.content)

