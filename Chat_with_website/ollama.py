import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import ollama
from langchain.text_splitter import CharacterTextSplitter as cts
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st


LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")

def main():
    st.title("Chat Bot Integrated inside website")
    st.subheader("Input your website URL, ask questions, and receive answers directly from the website")

    url = st.text_input("Insert Website URL")

    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
    )

    prompt_msg = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Insert your Prompt: {input}"),
    ]
    )   

    prompt = st.text_input(prompt_msg)

    if st.button("submit query", type="primary"):
        # Database dir created
        dir_name = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(dir_name,"db")

        loader = WebBaseLoader(url)
        data = loader.load()

        #Initialising of the text_splitter
        text_splitter = cts(separator= "\n", chunk_size= 1000, chunk_overlap= 50)

        docs = text_splitter.split_documents(data)

        # Creating ollama embeddings
        ollama_emb = OllamaEmbeddings(model = "mistral")

        # Creating Chrom DB
        vector_db = Chroma.from_documents(documents = docs, embedding = ollama_emb, persist_directory= db_path)

        vector_db.persist()

        retriever = vector_db.as_retriever(search_kwargs={'k': 5, 'fetch_k': 50})

        llm = ollama(model = mistral)
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        response = chain.invoke({"input": query})
        st.write(response)

if __name__ == "__main__":
    main()