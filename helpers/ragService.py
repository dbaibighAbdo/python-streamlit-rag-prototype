from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM


def langchain_expression_language(question: str, knowledge_base):

    # Prompt LCEL
    prompt = ChatPromptTemplate.from_template("""
        Vous êtes un assistant. Répondez uniquement en utilisant le contexte fourni. 
        Si le contexte n’est pas suffisant, répondez : « Je ne sais pas ».\n
        Contexte : {context}\n
        Question : {question}\n
    """)

    # LLM
    llm = OllamaLLM(
        model="llama3.2"
    )
    # Retriever
    retriever = knowledge_base.as_retriever()
    docs = retriever.invoke(question)

    # Build context text
    context_text = "\n\n".join([doc.page_content for doc in docs])
    print("context type:", type(context_text))
    print("question type:", type(question))


    # Build LCEL chain
    chain = (
        {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # IMPORTANT: execute chain and return result
    result = chain.invoke({"question": question, "context": context_text})
    return result