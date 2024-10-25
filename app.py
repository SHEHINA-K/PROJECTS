# Import necessary libraries and modules
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import hashlib
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key not found. Please check your .env file.")

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create and save the FAISS vector store
def get_conversational_chain():
    prompt_template = """
    Classify each customer support query into the appropriate intent category based on the context provided.
    Use the specified intent categories: `change_shipping_address`, `check_cancellation_fee`, `cancel_order`, and `cancel`.
    Follow the examples given and select the correct intent based on the customer's inquiry.
    If no intent matches, respond with "intent not available in the context".

    Context:
    - change_shipping_address: Queries about modifying the delivery address.
    - check_cancellation_fee: Requests for information on early termination or cancellation fees.
    - cancel_order: Requests related to canceling a specific order.
    - cancel: General questions on exiting or stopping a service.

    Examples:
    - "I have an issue changing the shipping address."
      Intent: `change_shipping_address`
    - "I don't know what I need to do to check the early exit fees."
      Intent: `check_cancellation_fee`
    - "Assistance to check the early termination penalty."
      Intent: `check_cancellation_fee`
    - "I cannot find the early exit penalties; can you help me?"
      Intent: `cancel`
    - "I need help canceling the last order I made."
      Intent: `cancel_order`
    - "How can I cancel my recent order?"
      Intent: `cancel_order`

    Query:
    {context}  # Use 'context' here

    Intent:
    """

    # Initialize the model and create the prompt
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])  # Use 'context'

    # Load the QA chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



# Function to handle user input and query the vector store
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

# Main function to handle the Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Customer Support Chatbot")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)  # Ensure this is called here
                    st.success("Processing completed!")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()

