from langchain_community.llms import Ollama 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
import tempfile


def pdf_rag(question, pdf_file):

    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name

    ## Loading the input pdf:
    pdf_loader = PyPDFLoader(temp_pdf_path)
    pdf_pages = pdf_loader.load_and_split()

    # Create a chunk splitter with 1000 chars each and 200 chars to overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the pages into docs based on the splits
    docs = text_splitter.split_documents(pdf_pages)

    model = Ollama(model="llama3")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorStore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="chroma_db")
    retriever = vectorStore.as_retriever()

    template = """
    You are an intelligent AI assistant who analyzes and answers question based on the given context. Answer the question based on the context provided. If you can't answer the question from the context, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    # Langchain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model 
        | parser
    )

    # Get the response
    return chain.invoke(question)


# # Sample usage
# question = "Who is charles's father?"
# pdf_path = "../sample_data/pdf/leclerc_sample.pdf"
# print(pdf_rag(question, pdf_path))