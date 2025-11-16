from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def langchain_expression_language(question: str, knowledge_base):

    # Prompt LCEL
    prompt = ChatPromptTemplate.from_template("""
        Vous êtes un assistant. Répondez uniquement en utilisant le contexte fourni. 
        Si le contexte n’est pas suffisant, répondez : « Je ne sais pas ».\n
        Contexte : {context}\n
        Question : {question}\n
    """)

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Retriever
    retriever = knowledge_base.as_retriever()
    docs = retriever.invoke(question)

    # Build context text
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Build LCEL chain
    chain = (
        {"question": question, "context": context_text}
        | prompt
        | llm
        | StrOutputParser()
    )

    # IMPORTANT: execute chain and return result
    result = chain.invoke({})
    return result