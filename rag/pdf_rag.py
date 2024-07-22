from langchain_community.llms import Ollama 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
# Rest of the code goes here

model = Ollama(model="llama3")
test_pdf_path = "./sample_data/pdf/sample.pdf"

parser = StrOutputParser()

template = """
You are an intelligent AI assistent curated to analyze PDF's. Answer the question based on the context provided below. If you can't 
answer the question from the given context, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
translation_prompt = ChatPromptTemplate.from_template("Translate {answer} to {language}")

chain = prompt | model | parser

translation_chain = ({"answer": chain, "language": itemgetter("language")} | translation_prompt | model | parser)

docs = pypdf.PDF(test_pdf_path)

for page in docs:
    text = RecursiveCharacterTextSplitter().split(page)
    chain.run(context=text, question="What is the main idea of this page?")
    print("Answer:", chain.output["answer"])
    translation_chain.run()
    print("Translated Answer:", translation_chain.output["answer"])
    print("\n")

