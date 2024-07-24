import toml
import streamlit as st
from streamlit_chat import message
import random
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.scrapper import ScrapeWebPage
from src.compressed_search import SimilarityCalculator
from src.vector_search import VectorSearch
from src.get_response import ResponseLLM
from src.ollama import OllamaGeneration
from PIL import Image

if 'messages' not in st.session_state:
    opener = 'Hi! Ask me anything regarding the contents of the URL. '
    st.session_state.messages = [{"role": "assistant", "content": opener}]
    st.session_state.content= []
    st.session_state.context = []
im = Image.open("./images/favicon.png")
st.set_page_config(page_title="suave.ai", page_icon=im)

with st.sidebar:
    st.image('images/mainlogo.png')
    new_title = '<p style="font-family: sans-serif; color:Black; font-size: 16px;">Transforming <b>conversations</b> into <b>conversions.</b></p>'
    url_page = '<b style="font-family: sans-serif; color:Black; font-size: 12px;"></b>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown("##")
    st.markdown("##")
    st.markdown(url_page, unsafe_allow_html=True)
    url = st.text_input("Please add a URL below to scrape your knowledge base.")
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    
    footer_info = '''<span style="font-family: Fantasy; color: black; font-size: 13px">
    We have all the pricing and informatione mentioned in our webpage.
    📖 Check out our <a href = "https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/">webpage</a>
    </span>'''
    st.markdown(footer_info, unsafe_allow_html=True)
    

@st.cache_data()
def scrape_url(url):
    url_scrapper = ScrapeWebPage(url)
    url_list, base_url = url_scrapper.get_url()
    processed_url = url_scrapper.process_urls(url_list=url_list, base_url=base_url)
    content = url_scrapper.get_page_contents(url_list = set(processed_url))
    vector_obj = VectorSearch(data=content, model_name="sentence-transformers/msmarco-distilbert-base-v3")
    return vector_obj

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
query = st.chat_input("Please add your query")

if url:
    with st.spinner("Scraping the webpage. Please wait."):
        vector_obj=scrape_url(url=url)


if query:
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs, metadatas = vector_obj._split_data()
            data_store = vector_obj._faiss_search()
            result = data_store.similarity_search(query)
            context = result[0].page_content
            answer_response = ResponseLLM(
                context=context,
                question=query,   
            )._generate()
            st.session_state.messages.append({"role": "user", "content": query})
            st.write(answer_response)
            st.write(result[0].metadata["source"])
            st.session_state.messages.append({"role": "assistant", "content": answer_response, "context": result[0].metadata["source"]})

