import os
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader

def load_documents_from_folder(folder_path):
    """Load all supported documents from a folder"""
    docs = []
    processed_files = []
    
    if not os.path.exists(folder_path):
        return [], []
        
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
            
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            
            
        docs.extend(loader.load())
    
    return docs, processed_files


def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """Split documents into chunks"""
    if not documents:
        return []
        
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    
    return chunks


def embed_and_store(chunks):
    """Create embeddings and store in FAISS vector database"""
    if not chunks:
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    vectorstore_dir = "rag_vectorstore"
    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    
    return vectorstore


def load_vectorstore():
    """Load a pre-existing vector store"""
    vectorstore_dir = "rag_vectorstore"
    
    if not os.path.exists(vectorstore_dir):
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
        
        return vectorstore
    except Exception as e:
        return None
    
    
def create_qa_chain(vectorstore, api_key="AIzaSyDQ5LZxY0TSqCHN2Be72vuK0I811jLYCsE"):
    """Create a question answering chain with the vector store"""
    if vectorstore is None:
        return None
        
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"k": 3, "score_threshold": 0.1}  
    )
    
    if not api_key:
        api_key = "AIzaSyDQ5LZxY0TSqCHN2Be72vuK0I811jLYCsE"
        return None
    
    # # Set the API key in the environment
    # os.environ["GOOGLE_API_KEY"] = api_key
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        return None

def initialize_rag_system(folder_path, chunk_size=500, chunk_overlap=100):
    """Initialize the RAG system by loading documents, chunking, embedding, and storing"""
    documents, processed_files = load_documents_from_folder(folder_path)
    
    if not documents:
        return None, None
    
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    
    vectorstore = embed_and_store(chunks)
    
    return vectorstore, processed_files

def main():
    folder_path = "story_books"
    chunk_size = 500
    chunk_overlap = 100
    
    # Initialize the RAG system
    # vectorstore, processed_files = initialize_rag_system(folder_path, chunk_size, chunk_overlap)
    # print(f"Processed {len(processed_files)} files from {folder_path}.")
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("No pre-existing vector store found. Initializing RAG system...")
        vectorstore, processed_files = initialize_rag_system(folder_path, chunk_size, chunk_overlap)
        print(f"Processed {len(processed_files)} files from {folder_path}.")
    
    
    # Create the QA chain
    qa_chain = create_qa_chain(vectorstore)
    
    if qa_chain is None:
        print("Failed to create QA chain.")
        return
    
    # Example query
    query = "Who is ALice?"
    response = qa_chain({"query": query})
    
    print("Response:", response['result'])
    print("Source Documents:", [doc.metadata for doc in response['source_documents']])
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")    