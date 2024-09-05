import ollama
import bs4 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def web_rag(question, url):
    loader = WebBaseLoader(
        web_path=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and vectorstore
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Combine documents into context
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)

    # Call the LLM with the retrieved context
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': f"Question: {question}\n\nContext: {formatted_context}"}])
    return response['message']['content']


# loader=WebBaseLoader(
#     web_path=("http://127.0.0.1:7860/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content","post-title","post-header")
#     )
# ),
# )

# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

# # 2. Create Ollama embeddings and vector store
# embeddings = OllamaEmbeddings(model="llama3")
# vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# # 3. Call Ollama Llama3 model
# def ollama_llm(question, context):
#     formatted_prompt = f"Question: {question}\n\nContext: {context}"
#     response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
#     return response['message']['content']

# # 4. RAG Setup
# retriever = vectorstore.as_retriever()
# def combine_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
# def rag_chain(question):
#     retrieved_docs = retriever.invoke(question)
#     formatted_context = combine_docs(retrieved_docs)
#     return ollama_llm(question, formatted_context)

# # 5. Use the RAG App
# result = rag_chain("What is Task Decomposition?")
# print(result)