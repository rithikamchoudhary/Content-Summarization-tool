import os
from langchain_community.llms import Anyscale
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import DuckDB
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from sklearn.metrics import precision_score, recall_score, f1_score

# Set API key
os.environ["ANYSCALE_API_KEY"] = "esecret_r1u6kcke1j42yfhmt1kjdh5pv1"

# Prompt template
template = """Question: {question}\nAnswer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

# Initialize Anyscale model
llm = Anyscale(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")
llm_chain = prompt | llm

# Load PDF document
pdf_path = r"C:\Users\frost\Desktop\python project\thebook.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split PDF document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
docs = text_splitter.split_documents(pages)

# Vectorize text chunks and store them in DuckDB
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = DuckDB(embedding=embedding_function)
vectorstore.add_documents(docs)

# Retrieve relevant documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Construct RetrievalQAWithSourcesChain
qachain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=retriever)

# Function to evaluate system performance
def evaluate(predictions, ground_truth):
    precision = precision_score(ground_truth, predictions, average='micro')
    recall = recall_score(ground_truth, predictions, average='micro')
    f1 = f1_score(ground_truth, predictions, average='micro')
    return precision, recall, f1

def load_evaluation_data():
    # Implement this function to load your evaluation dataset
    # Read your evaluation dataset from a file, database, or any other source
    # Return the loaded data, typically as a dictionary containing questions and answers
    evaluation_data = {
        "questions": ["what are the types of machine learning", "what is neural networks", ...],  # List of questions
        "answers": ["supervised, semi-supervised, unsupervised and reinforcement", "A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain. It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain.", ...]  # List of corresponding ground truth answers
    }
    return evaluation_data

# Heuristic evaluation function
def evaluate_heuristically(predicted_answers, ground_truth_answers):
    correct_count = 0
    total_count = len(predicted_answers)
    
    for pred_answer, true_answer in zip(predicted_answers, ground_truth_answers):
        if any(word.lower() in pred_answer.lower() for word in true_answer.split()):
            c
