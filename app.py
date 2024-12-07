import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
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
    Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details. If the answer is not in the provided context,
    just say, "Answer is not available in the context." Don't provide a wrong answer.
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def more_search(user_question):
    model = genai.GenerativeModel("gemini-pro")
    new_response = model.generate_content(user_question).text
    return new_response

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.session_state['pdf_response'] = response["output_text"]
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config("Chat with PDF", layout="wide")
    
    st.markdown("""
        <style>
        .stCodeBlock {
            max-width: 100% !important;
        }
        .stCode {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            max-width: 100% !important;
        }
        code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("Chat with PDFs using PDFPilot-Pro")

    if 'pdf_response' not in st.session_state:
        st.session_state['pdf_response'] = ""
    if 'additional_search_response' not in st.session_state:
        st.session_state['additional_search_response'] = ""

    with st.sidebar:
        st.title("Menu:")
        docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)
        if st.button("Submit"):
            if docs:
                with st.spinner("Processing..."):
                    raw_text = pdf_text(docs)
                    text_chunk = text_chunks(raw_text)
                    vector_store(text_chunk)
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files before submitting.")

    user_question = st.text_input("Ask Questions from the PDFs")
    if user_question:
        st.session_state['additional_search_response'] = ""
        
        st.subheader("Response from PDF content:")
        user_input(user_question)
        
        st.code(
            body=st.session_state['pdf_response'],
            language="text"
        )

        if st.button("Search beyond PDF content"):
            st.subheader("Additional search results:")
            st.session_state['additional_search_response'] = more_search(user_question)

        if st.session_state['additional_search_response']:
            st.code(
                body=st.session_state['additional_search_response'],
                language="text"
            )

if __name__ == "__main__":
    main()
