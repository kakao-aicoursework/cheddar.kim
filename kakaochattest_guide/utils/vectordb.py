
import chromadb
import openai
import os
import uuid
from pathlib import Path
from langchain.vectorstores import Chroma
from utils.const import *
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import (
    NotebookLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import CharacterTextSplitter


LOADER_DICT = {
    "py": TextLoader,
    "md": UnstructuredMarkdownLoader,
    "ipynb": NotebookLoader,
}
SPECIAL_FORMAT = ["txt"]

def create_vectordb () :
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print("vector db completed")
    return db


"""

카카오 데이터 포맷에 맞춰

category, title, type 으로 나누며 각각
category : 데이터의 기본 카테고리. 카카오싱크, 카카오소셜 등
title : 소제목. 개요, 기능 소개, 과정예시 등
type : 본문의 타입. text, table 등

으로 구성 됨

"""
def parse_markdown_manually (data_in_line:list) :
    
    meta_datas = []
    datas = []
    doc_title = ""
    tmp_data_name = ""
    tmp_data_context = ""
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    for i, v in enumerate(data_in_line) :
        meta_data = {"source":"local"}
        v = v.strip()
        # skip empty line
        if len(v) == 0 : continue
        # get title
        if i == 0 :
            doc_title = v
            continue
        
        if v[0] == "#" :
            # save previous data
            meta_data["category"] = doc_title
            meta_data["title"] = tmp_data_name
            meta_data["type"] = "text"
            
            if len(tmp_data_context) > 0 :
                # split text if it over chunk size
                splitted = splitter.split_text(tmp_data_context)
                for vv in splitted :
                    datas.append(vv)
                    meta_datas.append(meta_data)
            # set new data n keep going
            tmp_data_name = v[1:].strip()
            tmp_data_context = ""
            continue

        # accumulate context
        tmp_data_context += ("\n" + v)
    
    # save last data
    meta_data = {}
    meta_data["category"] = doc_title
    meta_data["title"] = tmp_data_name
    meta_data["type"] = "text"
    datas.append(tmp_data_context)
    meta_datas.append(meta_data)
    # remove trash
    datas.pop(0)
    meta_datas.pop(0)
    
    return datas, meta_datas

def parse_kakao_doc_txt (data_in_line:list) :
    
    datas, meta_datas = parse_markdown_manually(data_in_line=data_in_line)
    # for i in range(len(data[0])) :
    #     print(data[0][i], data[1][i])
    #     print("=--=-=-=-=-=-=")
    docs = [
        Document(page_content=data, metadata=meta_data) for data, meta_data in zip(datas, meta_datas)
        ]
    return docs

def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            ext = file.split(".")[-1]
            conds = [
                ext in SPECIAL_FORMAT,
                ext == "py",
                ext == "md",
                ext == "ipynb"]
            if any(conds) :
                file_path = os.path.join(root, file)

                try:
                    upload_embedding_from_file(file_path, ext)
                    print("SUCCESS: ", file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")
                    failed_upload_files.append(file_path)

def upload_embedding_from_file(file_path, ext):
    if ext == "txt" :
        documents = Path(file_path).open("r").readlines()
        documents = parse_kakao_doc_txt(documents)
    else :
        loader = LOADER_DICT.get(file_path.split(".")[-1])
        if loader is None:
            raise ValueError("Not supported file type")
        documents = loader(file_path).load()

    Chroma.from_documents(
        documents,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')

def read_embedding_from_file() :
    
    _db = Chroma.from_documents(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    _retriever = _db.as_retriever()
    print('db success')
    return _retriever

def query_kakao_suppl_info (query: str, db, use_retriever: bool = False) :
    if use_retriever:
        docs = db.get_relevant_documents(query, k=VDB_K)
    else:
        docs = db.similarity_search(query, k=VDB_K)

    str_docs = [doc.page_content for doc in docs]
    return str_docs
