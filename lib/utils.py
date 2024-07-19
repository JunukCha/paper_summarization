import os, os.path as osp

import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlretrieve
import zipfile

import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def extract_link(container, link_text, attribute='href'):
    link_element = container.find('a', string=lambda text: text and link_text in text.lower())
    if link_element:
        return link_element[attribute] if attribute != 'text' else link_element.get_text(strip=True)
    return 'Not available'

def scrape_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    papers = []
    for entry in soup.find_all('dt', class_='ptitle'):
        title = entry.text.strip()
        authors_dirty = entry.find_next_sibling('dd').text.strip()
        authors = ' '.join(author.strip() for author in authors_dirty.split('\n') if author.strip())
        links_container = entry.find_next_sibling('dd').find_next_sibling('dd')
        pdf_link = extract_link(links_container, 'pdf')
        supp_link = extract_link(links_container, 'supp')
        bibtex_info = links_container.find('div', class_='bibref').text if links_container.find('div', class_='bibref') else 'No BibTeX available'
        papers.append({
            'title': title,
            'authors': authors,
            'pdf_link': "https://openaccess.thecvf.com" + pdf_link,
            'supp_link': "https://openaccess.thecvf.com" + supp_link,
            'bibtex': bibtex_info
        })
    return papers

def save_to_excel(papers, save_path):
    df = pd.DataFrame(papers)
    counter = 1
    old_save_path = save_path
    while osp.exists(save_path):
        save_path = f"{old_save_path.rstrip('.xlsx')}_{counter}.xlsx"
        counter += 1
    df.to_excel(save_path, index=False)

def download_file(url, file_path):
    urlretrieve(url, file_path)
    if file_path[-4:] == ".zip":
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(file_path[:-4])

def save_markdown_to_file(md_content, md_file_path):
    md_content = md_content.replace("```markdown", "").replace("```", "")
    # Save Markdown content to a file
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

# Function to toggle the layout
def toggle_layout():
    if st.session_state.layout == 'Layout 1':
        st.session_state.layout = 'Layout 2'
        if 'papers' in st.session_state:
            del st.session_state.papers
    else:
        st.session_state.layout = 'Layout 1'
        if 'llm' in st.session_state:
            del st.session_state.llm
        if 'messages' in st.session_state:
            del st.session_state.messages
        if 'chat_history' in st.session_state:
            del st.session_state.chat_history

def stream_parser_rag(stream):
    for chunk in stream:
        if "answer" in chunk:
            yield chunk['answer']
            
def stream_parser_default(stream):
    for chunk in stream:
        yield chunk.content

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
    )
    loader = PyMuPDFLoader(file_path, extract_images=True)
    docs = loader.load()
    split_documents = text_splitter.split_documents(docs)

    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    faiss_vectorstore = FAISS.from_documents(split_documents, embedding=cached_embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever()
    return faiss_retriever, len(docs), len(split_documents)

def embed_multi_file(files, openai_api_key=None):
    documents = []
    
    store_name = None
    for file in files:
        if not osp.exists(file):
            continue
        with open(file, "rb") as f:
            file_content = f.read()
            file_path = f"./.cache/files/{f.name}"
            if store_name is None:
                if len(files) == 1:
                    store_name = f"./.cache/embeddings/{f.name}"
                elif len(files) >= 2:
                    store_name = f"./.cache/embeddings/{f.name}_w_supp"
        os.makedirs(osp.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)
        loader = PyMuPDFLoader(file_path, extract_images=True)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    split_documents = text_splitter.split_documents(documents)

    if openai_api_key is not None:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        store_name = osp.join(store_name, "openai")
    else:
        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        store_name = osp.join(store_name, "bge")
    cache_dir = LocalFileStore(store_name)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    faiss_vectorstore = FAISS.from_documents(split_documents, embedding=cached_embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever()
    return faiss_retriever

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


