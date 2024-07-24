from langchain.text_splitter import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
# from split_document import split_documents
from sentence_transformers import CrossEncoder
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

class VectorSearch:
    def __init__(self, data:list, model_name:str) -> None:
        self.data = data
        self.model_name = model_name
        
    
    def _split_data(self):
        
        text_splitter = CharacterTextSplitter(chunk_size=1500, separator='\n')
        self.docs, self.metadatas = [], []
        for page in self.data:
            splits = text_splitter.split_text(page['text'])
            self.docs.extend(splits)
            self.metadatas.extend([{"source": page['source']}] * len(splits))
        return self.docs, self.metadatas
    
    def _faiss_search(self):
        store = FAISS.from_texts(self.docs, self.embeddings, metadatas=self.metadatas)
        return store
    
    @staticmethod
    def _store_faiss(embedding_model_name, document):
        """Stores the splitted document according to toeknzier
            ARGS:
                embedding_model: str = name of model used for embedding 
                document: list = list of procesed markdown from the web.
        """
        embedding_model = HuggingFaceEmbeddings(
                            model_name=embedding_model_name,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
                        )

        knowledge_vector_database = FAISS.from_documents(
                                                        document, embedding_model, distance_strategy=DistanceStrategy.COSINE
                                                    )
        return knowledge_vector_database
    
    @staticmethod
    def _search(knowledge_vector_database, query):
        "Search for the query in the vector database."
        
        return knowledge_vector_database.similarity_search(query=query, k=5)  ## returns the 5 similar content from the vector store(knowledge vector database)
    
    @staticmethod
    def reranking_results(question, contexts):
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[question, doc.page_content ] for doc in contexts]
        scores = cross_encoder.predict(pairs)
        print(f"Scores of the embedding after reranking {scores}")
        return np.argsort(scores)[::-1][:2]
    




        
        
    
        
