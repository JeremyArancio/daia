from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from typing import *
from pathlib import Path

from PyPDF2 import PdfReader

import config


class DAIA():
    """Document AI Assistant to anwser any question relative to a document
    """

    docsearch: FAISS
    history: List[str]
    llm: OpenAI
    embeddings: OpenAIEmbeddings

    def __init__(
        self,
        open_ai_key: Optional[str] = None,
        **kwargs
    ) -> None:

        self.open_ai_key = open_ai_key
        self.history = []
        self.llm = OpenAI(
            openai_api_key=open_ai_key,
            model_name=config.text_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **kwargs
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=open_ai_key,
            query_model_name=config.embedding_model
        )
        self.docsearch: FAISS = None

    def process_document(self, file: Any) -> None:
        """Process the document to allow DAIA to search any particular information in it.
        Only plein text PDF for the moment.

        Args:
            filepath (Union[str, Path]): _description_
        """
        reader = PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages]) #TODO  Possibility to have access to pages
        text_splitter = SpacyTextSplitter(
            separator=config.separator, 
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        texts = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(openai_api_key=self.open_ai_key)
        self.docsearch = FAISS.from_texts(texts=texts, embedding=embeddings)

    def answer(self, question: str) -> str:
        """From a question relative to a document, generate the answer.

        Args:
            question (str): Question asked by the user.

        Returns:
            str: Answer generated with the LLM
        """
        qa = VectorDBQA.from_chain_type(
            llm=self.llm,
            chain_type=config.chain_type, 
            vectorstore=self.docsearch,
        )
        return qa.run(question)