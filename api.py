from pathlib import Path
import logging
import time
import pickle
from typing import *

import streamlit as st

from daia import DAIA


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
DATA_DIR = Path("examples/")


def process_uploaded_document() -> None:
    LOGGER.info(f"Start processing the document.")
    if st.session_state.uploaded_file:
        with st.spinner("The document is being processed..."):
            timestamp = time.time()
            st.session_state.daia.process_document(file=st.session_state.uploaded_file)
            LOGGER.info(f"It took {time.time() - timestamp} to process the document")

def import_existing_file() -> None:
    if st.session_state.existing_filename != "":
        LOGGER.info(f"Start importing existing file: {st.session_state.existing_filename}")
        with open(DATA_DIR / (st.session_state.existing_filename + ".docsearch"), 'rb') as f:
            docsearch = pickle.load(f)
            st.session_state.daia.docsearch = docsearch

def answer_question() -> str:
    LOGGER.info(f"Start answer module.")
    if st.session_state.question and st.session_state.daia.docsearch:
        with st.spinner("A moment, I'm looking at the document... It will take a few seconds."):
            LOGGER.info(f'The question is: {st.session_state.question}')
            st.session_state.answer = st.session_state.daia.answer(question=st.session_state.question)
            if "uploaded_file" in st.session_state:
                st.session_state.pages = st.session_state.daia.get_pages_from_sources(file=uploaded_file.getvalue())
    else:
        st.session_state.answer = "You have to give me a file to work with."
    

if "started" not in st.session_state:
    LOGGER.info("Starting the session.")
    st.session_state.daia = DAIA()
    st.session_state.started = True
    
st.title("DAIA: Document AI Assistant")

st.sidebar.markdown("Hello, I'm **DAIA**, your document AI assistant.")
st.sidebar.markdown("You can upload a PDF document and converse with me to obtain any answer. I love detailed questions. If you're not satisfied with my answers, complete your question with more details. **Have fun!**")
st.sidebar.markdown("**:yellow[IMPORTANT NOTE:]** *Don't put any sensitive documents.*")

# st.sidebar.text_input("Your open_ai_key", key="key")
# st.session_state.daia.openai_api_key = st.session_state.key

uploaded_file = st.sidebar.file_uploader("Input your file:", key='uploaded_file', on_change=process_uploaded_document)
st.sidebar.write('Or')
st.sidebar.selectbox(
    "Pick an example", 
    ["", "Miklagard IV Draft Slip 2022-23 GD", "proceedings-58-00031-v2"], 
    key="existing_filename",
    on_change=import_existing_file)

st.text_input(
    "What do you want to know about the document?", 
    key="question",
    on_change=answer_question
)

if "answer" in st.session_state:
    st.write(st.session_state.answer)
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.pages[0])
    with col2:
        st.image(st.session_state.pages[1])
    