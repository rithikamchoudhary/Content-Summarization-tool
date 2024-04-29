from langchain.chains import LLMChain
from langchain_community.llms import Anyscale
from langchain_core.prompts import PromptTemplate
#api key setup
import os

os.environ["ANYSCALE_API_KEY"] = "esecret_r1u6kcke1j42yfhmt1kjdh5pv1"
#promt
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
#calling the modell in
llm = Anyscale(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")
llm_chain = prompt | llm
#example promt
question = "When was George Washington president?"

result=llm_chain.invoke({"question": question})
#print(result)
#loading my pdf
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\frost\Desktop\python project\KRAKAUER JON - INTO THIN AIR.pdf")
pages = loader.load() 
len(pages)
#chunking the pdf 
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    separators=["\n\n", "\n", "\t"],
    chunk_size=10000,
    chunk_overlap=3000,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_documents(pages)
#vectorising and vector search and storing it in chroma db
from langchain.vectorstores import DuckDB
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
vectorstore = DuckDB(embedding=embedding_function)
#storing vectors and adding our docs
vectorstore.add_documents(docs)