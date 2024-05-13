from typing import List

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from functools import lru_cache

from langchain.llms import Anyscale
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


from dotenv import load_dotenv

load_dotenv()

llm = Anyscale(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",temperature=0,top_p=1)

def get_embeddings_transformer():
    """
    Returns the HuggingFaceInstructEmbeddings model for generating embeddings.
    """
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

@lru_cache(maxsize=1)
def get_vector_store():
    """
    Returns the Chroma vector store for storing and retrieving vectors.
    """
    vs = Chroma("ai_tutor", get_embeddings_transformer())
    return vs

@lru_cache(maxsize=1)
def get_mul_query_retriever():
    """
    Returns the MultiQueryRetriever that uses the vector store and LLM model for retrieval.
    """
    retriever = MultiQueryRetriever.from_llm(
        retriever=get_vector_store().as_retriever(search_kwargs={"k": 10}), llm=llm
    )
    return retriever

@lru_cache(maxsize=1)
def get_contextual_compression_retriever():
    """
    Returns the ContextualCompressionRetriever that uses the MultiQueryRetriever and LLMChainExtractor for retrieval.
    """
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 10})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever