from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import requests
import json
import os
import html2text
from langchain.embeddings import HuggingFaceEmbeddings
import openai

from langchain_core.prompts import PromptTemplate

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

urls = ["https://fastapi.tiangolo.com/", "https://fastapi.tiangolo.com/features/"]
# 1. Scrape raw HTML

def scrape_website(url):
    html_content = requests.get(url)
    return html_content.text

# 2. Convert html to markdown

def convert_html_to_markdown(html):

    # Create an html2text converter
    converter = html2text.HTML2Text()

    # Configure the converter
    converter.ignore_links = False

    # Convert the HTML to Markdown
    markdown = converter.handle(html)

    return markdown


def get_base_url(url):
    parsed_url = urlparse(url)

    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


# Turn relative url to absolute url in html

def convert_to_absolute_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')

    for img_tag in soup.find_all('img'):
        if img_tag.get('src'):
            src = img_tag.get('src')
            if src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['src'] = absolute_url
        elif img_tag.get('data-src'):
            src = img_tag.get('data-src')
            if src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['data-src'] = absolute_url

    for link_tag in soup.find_all('a'):
        href = link_tag.get('href')
        if href.startswith(('http://', 'https://')):
            continue
        absolute_url = urljoin(base_url, href)
        link_tag['href'] = absolute_url

    updated_html = str(soup)

    return updated_html


def get_markdown_from_url(url):
    # base_url = get_base_url(url)

    # print(f"Processing: {url}")
    html = scrape_website(url)
    markdown = convert_html_to_markdown(html)
    
    return markdown



from langchain.docstore.document import Document as LangchainDocument
from tqdm import tqdm

## step 1:  get content from the source
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=get_markdown_from_url(url=url), metadata={"source": url}) for url in tqdm(urls)
]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

EMBEDDING_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v3"

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

## step2 : tokenizer chunking
def split_documents(
    chunk_size: int,
    knowledge_base: LangchainDocument,
    tokenizer_name: str = EMBEDDING_MODEL_NAME,
):
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

docs_processed = split_documents(
    256,  # We choose a chunk size adapted to our model
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=EMBEDDING_MODEL_NAME,
)

print(docs_processed[10])

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)

KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)
# # print(docs_processed)

"""--------------------------------------------------query rewriting------------------------------------------"""

_standalone_prompt = """Given the following query 
###Question:
{question}
rephrase the question to be a standalone question, in its original language, 
that can be used to query a FAISS index. This query will be used to retrieve documents with additional context.
Give be the 5 rephrased question, format your answer in json object like.
'
  "0":"rephrased question 0",
  "1":"rephrased question 1",
  "2":"rephrased question 2",
  "3":"rephrased question 3",
  "4":"rephrased question 4",
'
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=_standalone_prompt,
)


user_query = "what is the dependency used for fastapi?"
print(f"\nStarting retrieval for {user_query=}...")

from langchain_openai import OpenAI
from langchain.chains import LLMChain


llm = OpenAI(openai_api_key="")
llm_chain = LLMChain(prompt=prompt, llm=llm)

svalues=llm_chain.invoke(input=user_query)


import json
json_values = json.loads(svalues["text"])

retrieved_docs = []

for i in range(len(json_values)):
  user_query = json_values["{}".format(i)]
  print(user_query)
  retrieved_docs += KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)


print("\n==================================Top document==================================")
print(retrieved_docs[3].page_content)
print("==================================Metadata==================================")
print(retrieved_docs[3].metadata)

## step3: Reranking 

print()
print("=================================Reranking documents ===================================")
from sentence_transformers import CrossEncoder

def reranking_results(question, contexts):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[question, doc.page_content ] for doc in contexts]
    scores = cross_encoder.predict(pairs)
    print(scores)
    return scores


import numpy as np
# Optionally rerank results
print("=> Reranking documents...")
scores = reranking_results(question=user_query, contexts=retrieved_docs)
context_num  = np.argsort(scores)[::-1][:2]
context1 = retrieved_docs[context_num[0]].page_content
context2 = retrieved_docs[context_num[1]].page_content

print(context1)
print()
print(context2)
print("=============================Generating Results=============================")

prompt_template = """ CONTEXT: {context}
    You are a helpful assistant, above is some context, 
    Please answer the question, and make sure you follow ALL of the rules below:
    1. Answer the questions only based on context provided, do not make things up
    2. Answer questions in a helpful manner that straight to the point, with clear structure & all relevant information that might help users answer the question

    QUESTION: {question}
    ANSWER (formatted in Markdown):
    
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

queries = {
    "question": user_query,
    "context": context1
}

## step 4 : REsult
print(llm_chain.invoke(input=queries))