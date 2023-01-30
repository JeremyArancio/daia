from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import VectorStore
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.callbacks import get_openai_callback

from typing import Optional, Union, Any, List
from pathlib import Path
import logging

from PyPDF2 import PdfReader
import pypdfium2 as pdfium

import config


LOGGER = logging.getLogger(__name__)


class DAIA():
    """Document AI Assistant to anwser any question relative to a document
    """

    docsearch: VectorStore
    history: List[str]
    llm: OpenAI
    embeddings: OpenAIEmbeddings
    sources: List[str]
    page_texts: List[str]
    reader: PdfReader

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
        self.docsearch: VectorStore = None

    def process_document(self, file: Union[Path, Any]) -> None:
        """Process the document to allow DAIA to search any particular information in it.
        Only plein text PDF for the moment.

        Args:
            file (Union[Path, Any]): file to process
        """
        self.reader = PdfReader(file)
        self.page_texts = [page.extract_text() for page in self.reader.pages]
        text = " ".join(self.page_texts)
        text_splitter = SpacyTextSplitter(
            separator=config.separator, 
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        texts = text_splitter.split_text(text)
        LOGGER.info(f'Number of chunks: {len(texts)}')

        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings(openai_api_key=self.open_ai_key)
            LOGGER.info(f"Number of tokens used for embeddings: {cb.total_tokens}")
            self.docsearch = FAISS.from_texts(
                texts=texts, 
                embedding=embeddings, 
                metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])

    def answer(self, question: str) -> str:
        """From a question relative to a document, generate the answer.

        Args:
            question (str): Question asked by the user.

        Returns:
            str: Answer generated with the LLM
        """
        with get_openai_callback() as cb:
            chain = VectorDBQA.from_chain_type(
                llm=self.llm,
                chain_type=config.chain_type, 
                vectorstore=self.docsearch,
                verbose=config.verbose
            )
            self.sources = self.get_sources(query=question) #  TODO: Already considered in VectorDBQAWithSourcesChain but impossible to retrieve
            answer = chain.run(question)
            LOGGER.info(f"Number of tokens used for answering: {cb.total_tokens}")
        return answer

    def get_sources(self, query: str, k: int = config.k) -> List[str]:
        """Retrieve chunks of text similar to the question

        Args:
            query (str): question asked by the user
            k (int, optional): Number of chunks returned. Defaults to config.k.

        Returns:
            List[str]: k chunks the most similar to the question.
        """
        docs = self.docsearch.similarity_search(query=query, k=k)
        return [doc.page_content for doc in docs]

    def get_pages_from_sources(
        self, 
        file: bytes, 
        n_char: int = config.n_char, 
    ) -> List[Any]:
        """From the chunk sources, return the pdf pages as images

        Args:
            file (bytes): uploaded file
            n_char (int, optional): Number of characters to consider to find the page of the chunk. Defaults to config.n_char.

        Returns:
            List[Any]: RGBA images supported by Streamlit
        """
        page_indices: List[int] = []
        images: List = []
        pdf = pdfium.PdfDocument(file)
        for source in self.sources:
            page_indices.extend([i for i, page in enumerate(self.reader.pages) if source[:n_char] in page.extract_text()])
        page_indices.sort()
        for i in page_indices:
            images.append(pdf.get_page(i).render_topil())
        return images