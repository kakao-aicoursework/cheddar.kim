from langchain import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import initialize_agent, Tool, AgentType
from utils.llm_template import get_langchain_template, read_prompt_template
from utils.vectordb import query_kakao_suppl_info
from utils.const import MAX_TOKEN, LLM_MODEL, TEMPLATES

def get_vector_from_openai (text) :
    embeddings_model = OpenAIEmbeddings()
    embedding = embeddings_model.embed_documents(texts=[text])[0]
    return embedding

def get_query_functions (vdb) :
    functions=[
        {
            "name" : "query_kakao_suppl_info",
            "func" : lambda x : query_kakao_suppl_info(x, vdb),
            "description" : "Function called when user asked about information of kakao information."
        }
    ]
    return functions

def get_single_query (text, vdb, temperature=0.0, model="gpt-3.5-turbo") :

    system_message_prompt = get_langchain_template("SYSTEM")

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])

    llm = ChatOpenAI(model=model, temperature=temperature)
    tools = []
    functions = get_query_functions(vdb)
    if functions != None :
        tools = [
            Tool(
                **func
            ) for func in functions
        ]
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        final_result = ""
        output = agent.invoke(text)
        final_result = output["output"]
        return {"answer" : final_result}
    else :
        agent = LLMChain(llm=llm, prompt=chat_prompt)
        return {"answer" : agent.run(text)}
    

def get_chained_query (text, vdb, temperature=0.0, model=LLM_MODEL) :
    
    def create_chain(llm, template_type, output_key, with_system=False):
        return LLMChain(
            llm=llm,
            prompt=get_langchain_template(template_type, with_system),
            output_key=output_key,
            verbose=True,
        )
    
    llm_00 = ChatOpenAI(temperature=temperature, max_tokens=MAX_TOKEN, model=model)
    llm_05 = ChatOpenAI(temperature=temperature, max_tokens=MAX_TOKEN, model=model)

    parse_intent_chain = create_chain(
        llm=llm_00,
        template_type="INTENT",
        output_key="intent",
    )
    category_chain = create_chain(
        llm=llm_00,
        template_type="CATEGORY",
        output_key="category",
    )
    kakao_answer_chain = create_chain(
        llm=llm_00,
        template_type="SUMMARIZE",
        output_key="output",
        with_system=True
    )
    # default_chain = ConversationChain(llm=llm_05, output_key="output")
    default_chain = create_chain(
        llm=llm_05,
        template_type="SYSTEM",
        output_key="output",
    )
    
    def gernerate_answer(user_message) :
        
        context = dict(user_message=user_message)
        
        context["input"] = context["user_message"]
        context["intent_list"] = read_prompt_template(TEMPLATES["INTENT_LIST"])

        # intent = parse_intent_chain(context)["intent"]
        intent = parse_intent_chain.run(context)
        print("intent", intent)

        if intent == "talk":
            answer = default_chain.run(context)
        elif intent == "question":
            answer = default_chain.run(context)
            
            # context["compressed_web_search_results"] = query_web_search(
            #     context["user_message"]
            # )
        
        elif intent == "question_kakao":
            context["category"] = category_chain.run(context)
            print("context[category]", context["category"])
            context["related_document"] = query_kakao_suppl_info(context["category"], vdb)
            answer = kakao_answer_chain.run(context)
        else :
            answer = default_chain.run(context)
            
        return {"answer": answer}
    
    return gernerate_answer(text)
