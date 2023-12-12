from dto import ChatbotRequest
from samples import list_card
import aiohttp
import time
import logging
import openai
import os
import requests
from utils.langchain import get_single_query

from utils.vectordb import create_vectordb, query_kakao_suppl_info

# 환경 변수 처리 필요!
openai.api_key = os.environ["LLM_LECTURE_KEY"]
os.environ["OPENAI_API_KEY"] = os.environ["LLM_LECTURE_KEY"]
SYSTEM_MSG = """
    당신은 카카오 서비스 제공자 봇입니다.
    응답은 한국어로 번역합니다.
"""
logger = logging.getLogger("Callback")

vdb_ids, vdb_docs, vdb_collection = create_vectordb()

# result = get_single_query("안녕하세요", SYSTEM_MSG, 
#                         functions=[
#                             {
#                                 "name" : "query_kakao_suppl_info",
#                                 "func" : lambda x : query_kakao_suppl_info(x, vdb_ids, vdb_docs, vdb_collection),
#                                 "description" : "유저가 카카오톡 싱크에 대해 추가 정보를 요청 할 때 부르는 함수."
#                             }
#                         ]
# )
# print("result", result)

async def callback_handler(request: ChatbotRequest) -> dict:

    query_functions = [
        {
            "name" : "query_kakao_suppl_info",
            "func" : lambda x : query_kakao_suppl_info(x, vdb_ids, vdb_docs, vdb_collection),
            "description" : "카카오톡 싱크에 대한 추가정보를 키워드를 받아 데이터베이스로부터 불러오는 함수."
        }
    ]
    # get a single data
    query_text = request.userRequest.utterance
    output_text = get_single_query(query_text, SYSTEM_MSG, functions=query_functions, temperature=0.0)
    print("output_text", output_text)
    
    # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": str(output_text).strip()
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    # time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()
    
    # print(request.userRequest)
    # if url:
    #     with requests.post(url=url, json=payload) as resp:
    #         print(resp.json())
