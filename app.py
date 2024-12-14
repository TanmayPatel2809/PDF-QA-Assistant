import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit setup
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # Manage chat session
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    # PDF file uploader moved to the sidebar
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Process uploaded PDFs and add spinner
    documents = []
    if uploaded_files:
        with st.spinner("Loading PDFs... Please wait."):
            for uploaded_file in uploaded_files:
                temp_pdf = f"./temp_{uploaded_file.name}"
                with open(temp_pdf, "wb") as file:
                    file.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temp_pdf)
                docs = loader.load()
                documents.extend(docs)

            # Split and create embeddings for documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
            retriever = vectorstore.as_retriever()

        # Enable the chat interface after PDF is loaded
        st.session_state.pdf_loaded = True
        st.success("PDFs loaded successfully!")

        # Contextualization prompt for history-aware retriever
        contextualize_q_system_prompt = (
            "You are a helpful assistant who has access to the user's uploaded documents. "
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "reformulate the question to make it clearer if necessary, without answering it. "
            "Your goal is to ensure the question is standalone and understandable even without the context. "
            "If no changes are needed, simply return the original question."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA system prompt
        system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer the question."
            "If you don't know the answer, say that you don't know."
            "Use Minimum sentences maximum and keep the answer concise."
            "\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to retrieve session history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User question input (enabled after PDF is loaded)
        if 'pdf_loaded' in st.session_state and st.session_state.pdf_loaded:
            user_input = st.text_input("Your question:")
            if user_input:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}} 
                )

                st.write("Assistant:", response['answer'])
                st.write("Chat History:", session_history.messages)

        else:
            st.warning("Please upload a PDF first.")
else:
    st.warning("Please enter the Groq API Key.")
