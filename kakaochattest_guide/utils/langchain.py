from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import SystemMessage


def get_vector_from_openai (text) :
    
    embeddings_model = OpenAIEmbeddings()
    embedding = embeddings_model.embed_documents(texts=[text])[0]
    return embedding

def get_single_query (text, system_msg, temperature=0.0, model="gpt-3.5-turbo", functions=None) :

    system_message_prompt = SystemMessage(content=system_msg)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])

    llm = ChatOpenAI(model=model, temperature=temperature)
    tools = []
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
        return final_result
    else :
        agent = LLMChain(llm=llm, prompt=chat_prompt)
        return agent.run(text)
    
