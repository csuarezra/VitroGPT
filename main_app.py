import streamlit as st
import os
import numpy as np

from typing import List
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, CSVLoader, UnstructuredWordDocumentLoader, EverNoteLoader, \
    UnstructuredEPubLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, UnstructuredODTLoader, PyMuPDFLoader, \
    UnstructuredPowerPointLoader, UnstructuredEmailLoader
from langchain.docstore.document import Document

from matplotlib import pyplot as plt
from PIL import Image

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

image = Image.open('images/ai_logo.png')
img_path = "img.jpg"

class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

# Page title
st.set_page_config(page_title='VitroGPT')
col1, col2 = st.columns([0.6, 0.4])
with col1:
    st.image(image, use_column_width="auto")
with col2:
    #st.title("VitroGPT")
    st.markdown("<h1 style='text-align: center;'>VitroGPT</h1>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and query_text:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, os.environ['OPENAI_API_KEY'], query_text)
            result.append(response)

if len(result):
    st.info(response)