import os, os.path as osp
import requests
import pandas as pd
import mimetypes
import threading
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import streamlit as st
import time
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ChatMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from lib.utils import print_messages, embed_file, format_docs, create_sources_string, stream_parser, retrieval_qa_chat_prompt, question_dictionary
from dotenv import load_dotenv
from docx.shared import Pt

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
        authors = ', '.join(author.strip() for author in authors_dirty.split('\n') if author.strip())
        links_container = entry.find_next_sibling('dd').find_next_sibling('dd')
        pdf_link = extract_link(links_container, 'pdf')
        supp_link = extract_link(links_container, 'supp')
        bibtex_info = links_container.find('div', class_='bibref').text if links_container.find('div', class_='bibref') else 'No BibTeX available'
        papers.append({
            'title': title,
            'authors': authors,
            'pdf_link': pdf_link,
            'supp_link': supp_link,
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

def save_markdown_to_file(md_content, md_file_path):
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

# Initialize session state to track layout
if 'layout' not in st.session_state:
    st.session_state.layout = 'Layout 1'

if 'running' not in st.session_state:
    st.session_state.running = False

st.set_page_config(page_title="ChatGPT", page_icon="😊")

# Layout 1
if st.session_state.layout == 'Layout 1':
    if 'run_scrape_button' in st.session_state \
        and st.session_state.run_scrape_button == True:
        st.session_state.running = True
    elif 'run_excel_button' in st.session_state \
        and st.session_state.run_excel_button == True:
        st.session_state.running = True
    elif 'run_pdf_button' in st.session_state \
        and st.session_state.run_pdf_button == True:
        st.session_state.running = True
    elif 'run_supp_button' in st.session_state \
        and st.session_state.run_supp_button == True:
        st.session_state.running = True
    elif 'run_summary_button' in st.session_state \
        and st.session_state.run_summary_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False

    # Button to switch layouts
    st.button("Chat", on_click=toggle_layout, disabled=st.session_state.running)

    st.title("Conference Paper Scraper")

    conference = st.selectbox("Conference", ["CVPR", "ICCV", "WACV"], disabled=st.session_state.running)
    if conference == "WACV":
        years = list(range(2024, 2019, -1))
    elif conference == "ICCV":
        years = list(range(2023, 2012, -2))  # ICCV is held every two years
    else:  # CVPR and other conferences
        years = list(range(2024, 2012, -1))
    year = st.selectbox("Year", years, disabled=st.session_state.running)
    query = st.text_input("Search Query", disabled=st.session_state.running)

    scrape_button = st.sidebar.button("Scrape Data", disabled=st.session_state.running, key='run_scrape_button')
    save_excel_button = st.sidebar.button("Save to Excel", disabled=st.session_state.running, key='run_excel_button')
    save_pdfs_button = st.sidebar.button("Save PDFs", disabled=st.session_state.running, key='run_pdf_button')
    save_supps_button = st.sidebar.button("Save Supps", disabled=st.session_state.running, key='run_supp_button')
    summary_button = st.sidebar.button("Summary", disabled=st.session_state.running, key='run_summary_button')

    if scrape_button:
        if query.strip():  # Check if query is not empty
            base_url = f"https://openaccess.thecvf.com/{conference}{year}"
            full_url = f"{base_url}?query={query}"
            papers = scrape_data(full_url)
            st.session_state['papers'] = papers
            st.rerun()  # Refresh the app to apply the enabled state
        else:
            st.session_state['none_query'] = True
            st.rerun()

    if 'none_query' in st.session_state:
        st.write("Query cannot be empty!")
        del st.session_state["none_query"]

    if summary_button:
        if query.strip():
            base_path = osp.join('material', conference, str(year), query)
            
            llm = ChatOllama(model="llama3", temperature=0)
            paper_list = os.listdir(base_path)
            
            with st.spinner():
                status_placeholder = st.empty()
                for paper_idx, paper in enumerate(paper_list):
                    st.session_state["chat_history"] = []
                    
                    with open(osp.join(base_path, paper, f"{paper}.pdf"), "rb") as file:
                        retriever, _, _ = embed_file(file)

                    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
                    history_aware_retriever = create_history_aware_retriever(
                        llm=llm, retriever=retriever, prompt=rephrase_prompt
                    )
                    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

                    qa = create_retrieval_chain(
                        retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
                    )

                    status_text = f"Summarizing: Question (0/{len(question_dictionary)}), Paper (1/{len(paper_list)})"
                    status_placeholder.text(status_text)
                    for l_subject_idx, (l_subject, m_subject_dict) in enumerate(question_dictionary.items()):
                        st.session_state["summary"] = []
                        
                        for m_subject, question in m_subject_dict.items():
                            output = qa.invoke(input={"input": question, "chat_history": st.session_state["chat_history"]})
                            st.session_state["chat_history"].append(("human", question))
                            st.session_state["chat_history"].append(("ai", output["answer"]))
                            st.session_state["summary"].append(("human", question))
                            st.session_state["summary"].append(("ai", output["answer"]))
                            
                        final_output = llm.invoke(f"Make it markdown: {st.session_state['summary']}")
                        base_name = f"summary_{l_subject.replace(' ', '_')}"
                        save_markdown_to_file(final_output.content, osp.join(base_path, paper, f"{base_name}.md"))
                        os.system(f'pandoc -s {osp.join(base_path, paper, f"{base_name}.md")} -o {osp.join(base_path, paper, f"{base_name}.docx")}')
                        
                        status_text = f"Summarizing: Question ({l_subject_idx+1}/{len(question_dictionary)}), Paper ({paper_idx+1}/{len(paper_list)})"
                        status_placeholder.text(status_text)

                st.rerun() # Refresh the app to apply the enabled state
        else:
            st.session_state['none_query'] = True
            st.rerun()
            
    if 'papers' in st.session_state:
        base_path = osp.join('material', conference, str(year), query)
        os.makedirs(base_path, exist_ok=True)
        
        papers = st.session_state['papers']
        st.write(f"Data scraped successfully! Number of papers: {len(papers)}")
        st.dataframe(pd.DataFrame(papers), height=200)
        
        if "saved_excel" in st.session_state:
            st.write(f"Data saved to Excel successfully!")
            del st.session_state["saved_excel"]
        if "saved_pdf" in st.session_state:
            st.write(f"PDF Files saved successfully!")
            del st.session_state["saved_pdf"]
        if "saved_supp" in st.session_state:
            st.write(f"Supp Files saved successfully!")
            del st.session_state["saved_supp"]

        if save_excel_button:
            save_path = osp.join(base_path, f"{conference}_{year}_{query}.xlsx")
            save_to_excel(st.session_state['papers'], save_path)
            st.session_state['saved_excel'] = True
            st.rerun()  # Refresh the app to apply the enabled state

        if save_pdfs_button or save_supps_button:
            if save_pdfs_button:
                link_type = "pdf_link"
                st.session_state['saved_pdf'] = True
            else:
                link_type = "supp_link"
                st.session_state['saved_supp'] = True
            
            st.write("Data are being downloaded...")
            papers = st.session_state["papers"]
            threads = []
            total_cnt = 0
            status_placeholder = st.empty()

            for paper in papers:
                url = paper[link_type]
                if url != "Not available":
                    folder_path = osp.join(base_path, paper['title'].replace(' ', '_'))
                    os.makedirs(folder_path, exist_ok=True)
                    if save_pdfs_button:
                        file_path = osp.join(folder_path, f"{paper['title'].replace(' ', '_')}.pdf")
                    else:
                        ext = mimetypes.guess_extension(mimetypes.guess_type(url)[0])
                        file_path = osp.join(folder_path, f"{paper['title'].replace(' ', '_')}_supp{ext}")
                    thread = threading.Thread(target=download_file, args=("https://openaccess.thecvf.com" + url, file_path))
                    threads.append(thread)
                    thread.start()
            for t_idx, thread in enumerate(threads):
                # status_placeholder.text()
                thread.join()
                if save_pdfs_button:
                    status_text = f"PDFs are being downloaded... ({t_idx+1}/{len(threads)})"
                else:
                    status_text = f"Supps are being downloaded... ({t_idx+1}/{len(threads)})"
                status_placeholder.text(status_text)
            st.rerun()  # Refresh the app to apply the enabled state
    else:
        if save_excel_button or save_pdfs_button or save_supps_button:
            st.rerun() # Refresh the app to apply the enabled state