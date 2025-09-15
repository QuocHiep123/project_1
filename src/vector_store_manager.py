# src/vector_store_manager.py

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# Em có thể cần import thêm os

def create_vector_store(chunks: List[Document]):
    '''
    Tạo ra một kho vector từ các chunks và lưu nó vào ổ đĩa.
    '''
    embedding_model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
    print('khởi tạo embedding')
    embeddings = HuggingFaceEmbeddings(model_name = embedding_model_name, model_kwargs = {'device' : 'cpu'})
    print('tạo kho vector từ chunks')
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store
