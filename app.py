import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def more_search(user_question):
    model = genai.GenerativeModel("gemini-pro")
    new_response = model.generate_content(user_question).text
    print(new_response)
    st.write("Reply:", new_response)   

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply from PDF content:", response["output_text"])    
    with st.expander("Click to expand and copy"):
        st.code(response["output_text"], language="text")  # Display with copy icon

def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Gemini")
    user_question = st.text_input("Ask Questions from the PDFs")
    flag=0
    if user_question:
        user_input(user_question)
        flag=1
        
    if flag==1:        
        st.write("Do you want to search beyond the PDF content?")
        additional_search = st.button("yes")
        if additional_search:
            more_search(user_question)
        flag=0

    with st.sidebar:
        st.title("Menu:")
        docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = pdf_text(docs)
                text_chunk = text_chunks(raw_text)
                vector_store(text_chunk)
                st.success("Done")

if __name__ == "__main__":
    main()
