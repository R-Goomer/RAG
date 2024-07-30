from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
import os


def __init__():
    # Load environment variables from .env file
    os.environ['OPENAI_API_KEY'] = os.getenv('open_ai_api')

def get_conversational_chain(retriever):
  
  prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
  """

  llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

  prompt = PromptTemplate(template = prompt_template, input_variables=["context","question"])

  ##chain = load_qa_chain(llm, chain_type = "stuff", prompt = prompt)

  chain = RetrievalQA.from_chain_type(
      llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
  )

  return chain

def llm_out(vec_store, user_question):

   ##retrieved_docs = vec_store.similarity_search(user_question)

    chain = get_conversational_chain(vec_store.as_retriever(search_type="similarity", search_kwargs={"k":2}))

    response = chain(
        {"query": user_question}, return_only_outputs = True
    )
    return response

  

