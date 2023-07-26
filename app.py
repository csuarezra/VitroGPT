import glob
import os
import shutil
import time
import typing
from typing import List
from PIL import Image

import streamlit as st
#from streamlit

from langchain.chains import ConversationChain, RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, CSVLoader, UnstructuredWordDocumentLoader, EverNoteLoader, \
    UnstructuredEPubLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, UnstructuredODTLoader, PyMuPDFLoader, \
    UnstructuredPowerPointLoader, UnstructuredEmailLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.docstore.document import Document


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


LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = []
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

title = "VitroGPT"
source_directory = "source_docs"
chunk_size = 1000
chunk_overlap = 0


PERSIST_DIRECTORY = "db_perist"
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)
persist_directory = PERSIST_DIRECTORY

theme = gr.themes.Base(
    radius_size=gr.themes.sizes.radius_sm,
)


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    results = []
    for file in filtered_files:
        print(f"Adding {file}")
        results.extend(load_single_document(file))
    return results


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(
                os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts






#This function will go through pdf and extract and return list of page texts.
def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        #print("Page Number:", len(pdfReader.pages))
        for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          text = pageObj.extract_text()
          pageObj.clear()
          text_list.append(text)
          sources_list.append(file.name + "_page_"+str(i))
    return [text_list,sources_list]
  
  
#file uploader
uploaded_files = st.file_uploader("Upload documents",accept_multiple_files=True, type=["txt","pdf"])
st.write("---")

if uploaded_files is None:
    st.info(f"""Upload files to analyse""")
elif uploaded_files:
    st.write(str(len(uploaded_files)) + " document(s) loaded..")
  
    textify_output = read_and_textify(uploaded_files)
  
    documents = textify_output[0]
    sources = textify_output[1]
  
    #extract embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
    #vstore with metadata. Here we will store page numbers.
    vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
    #deciding model
    model_name = "gpt-3.5-turbo"
    # model_name = "gpt-4"

    retriever = vStore.as_retriever()
    retriever.search_kwargs = {'k':2}

    #initiate model
    llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"], streaming=True)
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")
  
    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = model({"question":user_q}, return_only_outputs=True)
                st.subheader('Your response:')
                st.write(result['answer'])
                st.subheader('Source pages:')
                st.write(result['sources'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')




def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                               placeholder="Your Vitro AI assistant here! Ask me anything...",
                               label_visibility="hidden")
    
    return input_text







image = Image.open('vitro_logo.png')

col1, col2 = st.columns([0.35, 0.65])
with col1:
    st.image(image, width=200)
with col2:
    st.title("VitroGPT")


api_key = st.sidebar.text_input("OpenAI API-Key", key="openai_api_key", type="password")

uploaded_file = st.file_uploader("Upload Files")
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question and not api_key:
    st.info("Please add your Open API key to continue.")
