import os 
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

warnings.simplefilter("ignore")

dir_name = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(ABS_PATH, "db")

def create_vector_db():
    urls = ['https://docs.gpt4all.io/', 'https://ollama.com/library/llama2']

    url_loader = UnstructuredURLLoader(urls = urls, show_progress_bar = True)
    data = url_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 30)
    docs = text_splitter.split_documents(data)

    ollama_emb = OllamaEmbeddings(model = "mistral")

    vector_db = Chroma.from_documents(documents = docs, embedding = ollama_emb , persist_directory = db_path)
    vector_db.persist()




