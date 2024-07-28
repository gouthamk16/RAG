from langchain_community.llms import Ollama 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma
from langchain_community.embeddings import OllamaEmbeddings

## Loading the input pdf:
pdf_path = "../sample_data/pdf/sample.pdf"
pdf_loader = PyPDFLoader(pdf_path)
pdf_pages = pdf_loader.load_and_split()

model = Ollama(model="llama3")

parser = StrOutputParser()

template = """
You are an intelligent AI assistent curated to analyze PDF's. Answer the question based on the context provided below. If you can't 
answer the question from the given context, reply "I don't know".

Context: {context}

Question: {question}
"""

# translation_template = """
# Translate {answer} to {language}
# """

prompt = PromptTemplate.from_template(template)
## Adding some extra spice - answering in your local language
# translation_prompt = ChatPromptTemplate.from_template(translation_template)

chain = prompt | model | parser
## information on the chain -> chain.input_schema.schema()

# translation_chain = ({"answer": chain, "language": itemgetter("language")} | translation_prompt | model | parser)

## Initializing the vector store with a suitable embedding -> using a OllamaEmbeddings model for now
vectorStore = DocArrayInMemorySearch.from_documents(pdf_pages, embedding = OllamaEmbeddings(model="llama3"))
ret = vectorStore.as_retriever()

chain = (
    {"context": itemgetter("question") | ret, "question": itemgetter("question")}
    | prompt
    | model 
    | parser
)

chain.invoke("How does transformers work?")