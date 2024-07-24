import toml
import streamlit as st
from streamlit_chat import message
import random
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from src.scrapper import ScrapeWebPage
from src.compressed_search import SimilarityCalculator
from src.vector_search import VectorSearch
from src.get_response import ResponseLLM
from src.ollama import OllamaGeneration
from src.split_document import split_documents
from src.generate_rephrased_question import paraphrase_question
from langchain.docstore.document import Document as LangchainDocument
from src.add_image_markdown import get_content
from tqdm import tqdm
import json


if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.contect= []

st.set_page_config(page_title=" AI", page_icon="🌐")
with st.sidebar:
    st.title('🦙💬 AI')
    st.write('AI That reads and answer question based ..')

    url = st.text_input("Please add a URL: ")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    # top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    # max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown('📖 Check out our webpage [blog]!')
  

@st.cache_data()
def scrape_url(url):
    url_scrapper = ScrapeWebPage(url)
    url_list, base_url = url_scrapper.get_url()
    processed_url = url_scrapper.process_urls(url_list=url_list, base_url=base_url)
    RAW_KNOWLEDGE_BASE = [
                            LangchainDocument(page_content=get_content(url=url), metadata={"source": url}) for url in tqdm(processed_url)
                            ]
    docs_processed = split_documents(RAW_KNOWLEDGE_BASE, 256, "sentence-transformers/msmarco-distilbert-base-v3")  ## keeping in cache so it doesnot split multiple times. 
    
    embedding_model = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/msmarco-distilbert-base-v3",
                                model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity

                                                        )
    knowledge_vector_database = FAISS.from_documents(docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    return knowledge_vector_database

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("`Ask a question:`")

if url:
    with st.spinner("Scraping the webpage. Please wait."):
        knowledge_vector_database=scrape_url(url=url)


if query:
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            rephrased_questions_response_json = paraphrase_question(question=query)
            
            retrieved_docs = []
            for i in range(len(rephrased_questions_response_json)):
                user_query = rephrased_questions_response_json["{}".format(i)]
                print(user_query)
                retrieved_docs += knowledge_vector_database.similarity_search(query=user_query, k=5)
            
            # print(retrieved_docs)
            top_similar_ranked_content = VectorSearch.reranking_results(
                    question=user_query,
                    contexts=retrieved_docs
                )
            result = retrieved_docs[top_similar_ranked_content[0]]
            print(result)
            # context = retrieved_docs[top_similar_ranked_content[1]]
            # print(context)
            context = result.page_content
            answer_response = ResponseLLM(
                context=context,
                question=query,   
            ).generate_markdown()
            st.session_state.messages.append({"role": "user", "content": query, "context": context})
            st.markdown(answer_response)
            st.write(result.metadata["source"])
            st.session_state.messages.append({"role": "assistant", "content": answer_response, "context": context})

