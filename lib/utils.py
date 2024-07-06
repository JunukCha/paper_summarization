import os, os.path as osp

import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def stream_parser(stream):
    for chunk in stream:
        if "answer" in chunk:
            yield chunk['answer']

def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    os.makedirs(osp.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        # separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        # length_function=len,
    )
    # loader = UnstructuredFileLoader(file_path)
    loader = PyMuPDFLoader(file_path, extract_images=True)
    docs = loader.load()
    # docs = loader.load_and_split(text_splitter=text_splitter)
    split_documents = text_splitter.split_documents(docs)

    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    # embeddings = BGEM3FlagModel('BAAI/bge-m3',  
    #                    use_fp16=True)
    # embeddings = OpenAIEmbeddings()
    # embeddings = OllamaEmbeddings(model="llama3")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir, namespace=embeddings.model)
    bm25_retriever = BM25Retriever.from_documents(split_documents)

    faiss_vectorstore = FAISS.from_documents(split_documents, embedding=cached_embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever()
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4],
        search_type="mmr",
    )
    # retriever = ensemble_retriever.as_retriever()
    return ensemble_retriever, len(docs), len(split_documents)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_sources_string(source_urls):
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


retrieval_qa_chat_prompt_template="""
SYSTEM

You are an expert in the field of artificial intelligence and the first author of the provided paper. Your role is to thoroughly review the contents of the provided files and answer the questions I ask, based on the provided files, to a graduate student majoring in artificial intelligence. All answers must be based on the provided content. Do not provide false answers, and if you do not know the answer, please say so.
Please carefully review and summarize the provided paper as a whole.
If you write a formula, please write it in latex format. Be sure to include the title and source of the reference. The style of the references is IEEE style as shown below.
Example references:
A. Author, B. Author, and C. Author, "Title of the paper," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Year

<context>

{context}

</context>

PLACEHOLDER

chat_history
HUMAN

{input}
"""
retrieval_qa_chat_prompt = PromptTemplate.from_template(retrieval_qa_chat_prompt_template)

question_dictionary = {
    "Introduction": {
        "Background": "Please provide a rich and detailed explanation of what was the main motivation for starting this research and what problem or challenge you were trying to solve.",
        "Contribution": "Could you please provide a rich and detailed description of the main contributions of this paper?"
    },

    "Related Works": {
        "Baseline": "I would greatly appreciate it if you could tell me in detail what the Baseline is on the basis of the methodology explained in the paper, and what structure or paper was referenced for the Baseline of the paper. If there is no Baseline, please tell me there is none.",
        "Related Works": "Can you introduce key studies that are closely related to the task or method covered in the paper? Please respond based on the contents of the paper."
    },

    "Methodology": {
        "Preliminaries": """Can you please tell me some important references or prerequisite concepts that I need to know to fully understand the Method? I would greatly appreciate it if you could explain these references or concepts in a simple and easy-to-understand manner. If necessary, please refer to Wikipedia. , arxiv, etc. and let me know.
If you write a formula, please write it in latex format. Be sure to include the title and source of the reference. The style of the references is IEEE style as shown below.

Example references:
A. Author, B. Author, and C. Author, "Title of the paper," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Year""",

        "Backbone": "I would greatly appreciate it if you could tell me in detail and in detail what the backbone network of the methodology described in the paper is and what structure or paper the backbone network of the paper was referenced.",
        "Methodology": "Could you provide a rich and detailed explanation of the main methodology (Method) used in the paper? I would really appreciate it if you could break down the methodology into a step-by-step process and explain it clearly and in an easy-to-understand manner.",
        "Figure": "I would appreciate it if you could tell me the location of the figure representing the methodology model or framework architecture based on the paper and explain the figure.",
        "Loss function": "Could you provide a detailed explanation of the loss function or formulas used in this paper? Please explain the purpose and function of these loss functions in detail and in an easy-to-understand manner."
    },

    "Experiments": {
        "Datasets": 'Could you provide a comprehensive and detailed list of all the datasets used in this paper? I would greatly appreciate it if you could explain in detail the characteristics and role of each dataset. The characteristics of the dataset include what data it contains. "What has been collected and how much is collected? The role of the dataset includes training purposes, quantitative evaluation, and application.',
        "Implementation Details": "Can you comprehensively describe the implementation details and experiment setup used during the training process? In particular, the number and type of GPUs used to train the model, and the training period. I would also like to know the specific hyperparameters.",
        "Quantitative Metrics": """Can you provide a detailed description of the specific quantitative metrics used in this paper? Please explain in detail how these metrics were developed and how they should be interpreted. Wikipedia, if necessary, Please search arxiv etc. and let me know.

If you write a formula, please write it in latex format. Be sure to include the title and source of the reference. The style of the references is IEEE style as shown below.

Example references:
A. Author, B. Author, and C. Author, "Title of the paper," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Year""",

        "Quantitative Result": "Could you please provide a comprehensive analysis of the results achieved by the methodology presented in this paper, especially with regard to the aforementioned datasets and quantitative metrics? The strengths and weaknesses of this methodology I would greatly appreciate it if you could briefly explain and provide an explanation as to why this result was reached.",
        "Qualitative Result": "Could you provide a comprehensive description and analysis of the qualitative results presented in the paper? Additionally, could you specifically tell me which figures and which sections I should refer to to fully understand these qualitative results? ",
        "Ablation Study": "Can you provide a comprehensive description of the ablation study mentioned in the paper? Please provide a detailed description of the ablation study, including the purpose, methodology, and results of the study. Also, the importance and meaning of the ablation study results. Please let us know clearly.",
        "Meaning of Result": "Can you please provide a detailed and systematic explanation of the meaning and importance of the results presented in the paper, thoroughly analyzed and interpreted, taking into account all research results and their implications answered so far?"
    },

    "Conclusion": {
        "Limitation": "Please provide a comprehensive and detailed explanation of the limitations of the method presented in this paper. Also, please discuss any restrictions or limitations that may arise when implementing it.",
        "Application": "Please explain in detail how the aforementioned limitations can be overcome and how this methodology can be combined with other AI/ML methodologies. As an artificial intelligence expert, please feel free to imagine.",
        "Summary": "Please write a rich, structured summary of the paper, taking into account what we have discussed in the conversation so far. Your response should be detailed and comprehensive, identifying the paper's main findings, arguments, and contributions."
    },

    "Add to read":{
        "Future Work": "Please find subsequent papers that cite this paper. Please also provide a brief introduction to each paper.",
        "State of the arts": "Please search and find the most recent state of the arts papers on the task covered in this paper. Please also provide a brief introduction to each paper."
    }
}