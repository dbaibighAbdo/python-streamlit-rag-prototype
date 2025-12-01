from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import os 
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("OLLAMA_API_KEY")


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_text(text)
    return chunks

def embed_store_texts(chunks):
    '''
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    '''
    embedding_model = OllamaEmbeddings( 
        model="mxbai-embed-large"
    )
    knowledge_base = FAISS.from_texts(chunks, embedding_model)
    knowledge_base.save_local("FAISS_DB/faiss_index")
    return knowledge_base

def process_pdf(file):
    text = extract_text_from_pdf(file)
    chunks = split_text(text)
    knowledge_base = embed_store_texts(chunks)
    return knowledge_base

