import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter
load_dotenv()

print("Initializing components...")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = GoogleGenerativeAI(model="models/gemini-3-flash-preview")
vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv('INDEX_NAME'))

retriver = vectorstore.as_retriever(search_kwargs={"k": 3})

promt_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, say you don't know.

    {context}

    Question: {question}

    Provide a detailed answer:
    """,

)

def format_docs(docs):
    """Format the retrieved documents into a single string """ 
    
    return "\n\n".join(doc.page_content for doc in docs)

def retrival_chain_without_lcel(query:str):
    
    """A simple retrival chain without LCEl""" 
    retrival_documents = retriver.invoke(query)
    context = format_docs(retrival_documents)
    prompt = promt_template.format_messages(context=context, question=query)
    response = llm.invoke(prompt)
    return response

def retrival_chain_with_lcel():
    """A simple retrival chain with LCEl""" 
    retrival_chain =  ( RunnablePassthrough.assign(context= itemgetter("question")| retriver | format_docs) 
                       | promt_template 
                       | llm 
                       | StrOutputParser()) 

    return retrival_chain


if __name__ == "__main__":
    print("Retriving...")

    #Query
    question = "What is the Pinecone in machine learning?"

    # ===================================================
    # RAW  invocation without RAG
    # ====================================================

    print ("\n" + "=" * 70)
    print("RAW invocation without RAG")
    print("=" * 70 + "\n")
    result_raw = llm.invoke([HumanMessage(content=question)])
    print("Answer without RAG:")
    print(result_raw)

    print ("\n" + "=" * 70)
    print("RAW invocation with RAG")
    print("=" * 70 + "\n")
    result_with_rag = retrival_chain_without_lcel(question)
    print("Answer with RAG:")
    print(result_with_rag)

    print ("\n" + "=" * 70)
    print("RAW invocation with RAG and LCEL")
    print("=" * 70 + "\n")
    chain_with_lcel = retrival_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": question})
    print("Answer with RAG and LCEL:")
    print(result_with_lcel)
