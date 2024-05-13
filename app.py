import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader

from langchain.llms import Anyscale
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from retrivers import get_contextual_compression_retriever, get_embeddings_transformer, get_vector_store, get_mul_query_retriever
from langchain.prompts import PromptTemplate

import hashlib
from tqdm import tqdm
from functools import lru_cache

import os
from dotenv import load_dotenv

load_dotenv()

one_way_hash = lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()

PROMPT = PromptTemplate(template="""
       ##Persona: A student who is studying for an exam and needs help with a question.
       You are an AI Tutor Bot; Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
       {context}
       ##Question:{question} \n\
       ##AI Assistant Response:\n""",input_variables=["context","question"])

def get_pdf_data(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t"],
        chunk_size=1000,
        chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

@lru_cache(maxsize=1)
def get_conversation_chain():
    llm = Anyscale(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",temperature=0,top_p=1)
    retriever = get_mul_query_retriever()
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

def handle_userinput(user_question):
    conv = st.session_state.conv
    response = conv({'query': user_question})
    chat_history = [{"user": response["query"], "assistant": response["result"]}]
    st.session_state.chat_history += chat_history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["assistant"])
        # if i % 2 == 0:
        #     with st.chat_message("user"):
        #         st.markdown(message)
        # else:
        #     with st.chat_message("assistant"):
        #         st.markdown(message.content)



st.set_page_config(page_title="Chat with multiple PDFs",
                    page_icon=":books:")
st.session_state.vectorstore = get_vector_store()
st.session_state.conv = get_conversation_chain()
tab1, tab2 = st.tabs(["Q & A", "ADD document"])
with tab1:        
    st.title("""Assistant AI Tutor""")

    chat_history_clear = st.button("Clear Chat History")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if ("chat_history" not in st.session_state) or chat_history_clear:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        print(">>>>>>>>>>>>>>>")
        handle_userinput(user_question)


with tab2:
    st.subheader("Your documents")
    pdf = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=False)
    b = st.button("Process")
    if b:
        vs = st.session_state.vectorstore
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_data(pdf)
            if raw_text:
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                if len(text_chunks)>500:
                    split = 100
                else:
                    split = 10
                for i in range(0, len(text_chunks), split):
                    batch_chunks = text_chunks[i:(i + split-1)]
                    vs.add_texts(batch_chunks)
                st.write('Document added successfully')

    with st.sidebar:
        add_vertical_space(3)
        st.title("Process your PDFs and perform vector search")
        st.markdown('''
        ## About
        This app is an LLM-powered AI tutor that can help with:
        - Answering questions from your documents
        - Providing explanations for the answers
        - Providing additional context for the answers
        ''')
        add_vertical_space(5)
        st.write('Made with ❤️ by [Your Name](<your linkedin profile url>) and [Your Name 2 ](your linkedin profile url)')