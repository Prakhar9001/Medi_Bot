from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import PyPDFLoader  
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
import shutil  
import os  

# Data path to get information from  
DATA_PATH = r"C:\Users\mehul\edumit\llama2-PDF-Chatbot\data\Medical_book (2).pdf"  
# Path to store the embeddings  
DB_FAISS_PATH = r"C:\Users\mehul\edumit\llama2-PDF-Chatbot\vectorstores\db_faiss"  

# Create a vector database  
def create_vector_db():  
    # Check if user wants to override vectorstores with new data or not  
    if os.path.exists(DB_FAISS_PATH):  
        overwrite = input("Vector store already exists do you want to override? (y/n): ")  
        if overwrite.lower() == 'y':  
            shutil.rmtree(DB_FAISS_PATH)  
            os.makedirs(DB_FAISS_PATH)  # Recreate the directory  
        else:  
            print("Vector store were not overridden. Exiting...")  
            return  
    else:  
        os.makedirs(DB_FAISS_PATH)  # Create directory if it doesn't exist  
    
    # Load PDF file using PyPDFLoader directly  
    loader = PyPDFLoader(DATA_PATH)  
    documents = loader.load()  # Load documents as text chunks  
        
    # Split text chunks into smaller segments  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)  
    texts = text_splitter.split_documents(documents)  
    
    # Create text embeddings; numerical vectors that represent the semantics of the text  
    try:  
        embeddings = HuggingFaceEmbeddings(  
            model_name="sentence-transformers/all-MiniLM-L6-v2",   
            model_kwargs={'device': 'cuda'}  
        )  
    except:  
        # Fallback to CPU if CUDA is not available  
        embeddings = HuggingFaceEmbeddings(  
            model_name="sentence-transformers/all-MiniLM-L6-v2"  
        )  
    
    # Create a vector store using the embeddings and save it locally  
    db = FAISS.from_documents(texts, embeddings)  
    db.save_local(DB_FAISS_PATH)  
    print("Vector database created successfully!")  

if __name__ == "__main__":  
    create_vector_db()