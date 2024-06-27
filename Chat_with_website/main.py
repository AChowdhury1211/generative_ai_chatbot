import os

from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import ollama
from langchain.callbacks.manager import CallbackManager

