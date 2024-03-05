import os
import tempfile

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from components.sidebar import sidebar
from core.chunking import chunk_file
from core.parsing import read_file
from ui import display_file_read_error, is_open_ai_key_valid, is_query_valid

MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

st.set_page_config(page_title="Advanced RAG DDAM", page_icon="📖", layout="wide")
st.header("📖Advanced RAG DDAM")

openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

if not openai_api_key:
    st.warning(
        "Please export your OpenAI API key in command line \n"
        "EXPORT OPENAI_API_KEY='sk-**************'\n"
        "You can get a key at\n"
        "https://platform.openai.com/account/api-keys"
    )

if not cohere_api_key:
    st.warning(
        "Please export your Cohere API key in command line to your environment variables\n"
        "EXPORT COHERE_API_KEY='**************'\n"
        "You can get a key at\n"
        "https://dashboard.cohere.com/api-keys"
    )

uploaded_files = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf"],
    help="Please upload your pdf(s)",
    accept_multiple_files=True,
)

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

if not uploaded_files:
    st.stop()

for i in uploaded_files:
    try:
        file = read_file(i)
    except Exception as e:
        display_file_read_error(e, file_name=i.name)


if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()


temp_dir = tempfile.mkdtemp()
uploaded_file_paths = []
for file in uploaded_files:
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.getvalue())
    uploaded_file_paths.append(path)

if uploaded_file_paths:
    chunks = chunk_file(uploaded_file_paths)

if not chunks:
    st.stop()

with st.spinner("Generating response, please wait!"):
    embedding_model = OpenAIEmbeddings(
        api_key=openai_api_key,  # type:ignore
        model="text-embedding-ada-002",  # type:ignore
    )
    db = Chroma.from_documents(chunks, embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )


with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 

    Question: {question} 

    Context: {context} 
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  # type:ignore
        model=model,
        temperature=0,
    )
    rag_chain = (
        {
            "context": compression_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    retrieved_context = compression_retriever.get_relevant_documents(query)
    # retrieved_context = [x.page_content for x in retrieved_context]

    llm_response = rag_chain.invoke(query)

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(llm_response)

    with sources_col:
        st.markdown("#### Sources")
        for source in retrieved_context:
            st.markdown(source.page_content)
            st.markdown("---")
