
TEMPLATES = {
    "SYSTEM" : "llm_templates/system.txt",
    "INTENT" : "llm_templates/intent.txt",
    "INTENT_LIST" : "llm_templates/intent_list.txt",
    "CATEGORY" : "llm_templates/kakao_cat.txt",
    "SUMMARIZE" : "llm_templates/summarize_info.txt"
}

CHROMA_PERSIST_DIR = "datas/chroma_persist"
CHROMA_COLLECTION_NAME = "kakao_api"

MAX_TOKEN = 500
LLM_MODEL = "gpt-3.5-turbo"

VDB_K = 2