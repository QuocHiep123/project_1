# src/data_processor.py

from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

# Em có thể cần import thêm các thư viện khác như os

def load_documents(file_path: str) -> List[Document]:
    """
    Đọc tất cả các file .txt và .pdf từ một thư mục được chỉ định.

    Args:
        folder_path (str): Đường dẫn đến thư mục chứa dữ liệu (ví dụ: './data').

    Returns:
        List[dict]: Một danh sách các tài liệu đã được load.
    """
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            print(f'Không hỗ trợ định dạng này {file_path}')
            return []
        return loader.load()
    except Exception as e:
        print(f'Lỗi {file_path} : {e}')
        return []

def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    '''
    Cắt các tài liệu lớn thành các đoạn nhỏ (chunks).
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap= chunk_overlap,
        length_function= len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

