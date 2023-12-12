
import chromadb
import openai
# import markdown_to_json
import json
import uuid
from pathlib import Path
from utils.langchain import get_vector_from_openai
import time

def create_vectordb (persistant=False, distance_metric="l2") :
    
    # create vectordb client
    client = chromadb.PersistentClient() if persistant else chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="kakao_api",
        metadata={"hnsw:space": distance_metric}
    )
    vectordb_ids, vectordb_docs = data_vectorization_using_gpt(collection)
    
    print("vector db completed")
    
    return vectordb_ids, vectordb_docs, collection

def load_kakao_text_info (text_name):
    file_path = Path(f"datas/project_data_{text_name}.txt")
    if file_path.is_file() :
        return file_path.open("r", encoding="utf-8").read()
    else :
        raise f"file {file_path.absolute()} erorr"

def parse_markdown_manually (data_in_line:list) :
    
    data = {}
    tmp_data_name = ""
    tmp_data_context = ""
    
    for i, v in enumerate(data_in_line) :
        if len(v) == 0 : continue
        if i == 0 :
            tmp_data_name = "doc_title"
            tmp_data_context = v
            continue
        if v[0] == "#" :
            data[tmp_data_name] = tmp_data_context.strip()
            tmp_data_context = ""
            tmp_data_name = v[1:].strip()
            continue
        tmp_data_context += "\n"
        tmp_data_context += v
        continue
    
    return data

def data_vectorization_using_gpt (collection) :
    
    temperature = 0.0
    max_tokens = 4096
    
    data = load_kakao_text_info("kakao_sync")
    data = "\n".join([v.strip() for v in data.split("\n")])
    
    # jsonified = markdown_to_json.jsonify(data)
    
    parsed_data = parse_markdown_manually(data.split("\n"))
    # print("data", parsed_data)
    
    ids = []
    doc_meta = []
    documents = {}
    embeddings = []

    doc_title = ""
    for i, key in enumerate(parsed_data.keys()) :
        if key == "doc_title" :
            doc_title = parsed_data[key]
            continue
        id = str(uuid.uuid4())[:8]

        document_to_embed = f"{key}: {parsed_data[key]}"

        meta = {
        }
        embedding = get_vector_from_openai(document_to_embed)

        ids.append(id)
        doc_meta.append(meta)
        documents[id] = {
            "Doc" : doc_title,
            "Title" : key,
            "Contents" : parsed_data[key],
        }
        embeddings.append(embedding)
    
    # DB 저장
    collection.add(
        # documents=documents,
        embeddings=embeddings,
        # metadatas=doc_meta,
        ids=ids
    )
    
    return ids, documents

def query_kakao_suppl_info (query_text, vectordb_ids, vectordb_docs, collection) :
    
    result = collection.query(
        query_embeddings=get_vector_from_openai(query_text),
        n_results=2,
    )
    
    results = []
    for id in result["ids"][0] :
        results.append(vectordb_docs[id])
    return results
