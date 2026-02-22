import streamlit as st
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings, DatabricksVectorSearch, VectorSearchRetrieverTool
from databricks.sdk import WorkspaceClient
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate

llm_model = "databricks-gemma-3-12b"

embedding_model = "databricks-bge-large-en"

vector_search_endpoint = "test_vector_search"

catalog_name = "workspace"
schema_name = "my_schema"
table_doc_embedding = f"{catalog_name}.{schema_name}.rag_doc_embedding"

vector_store = DatabricksVectorSearch(
    endpoint=vector_search_endpoint,
    index_name=table_doc_embedding,
    # If using chunk_vector column manually, then enable 'embedding' and 'text_column' parameters.
    # The index 'workspace.my_schema.rag_doc_embedding' uses Databricks-managed embeddings. Do not pass the `embedding` parameter when initializing vector store. 
    # embedding=DatabricksEmbeddings(endpoint=embedding_model),
    # The index 'workspace.my_schema.rag_doc_embedding' has the source column configured as 'chunk_text'. Do not pass the `text_column` parameter.
    # text_column="chunk_text"
)

retriever = vector_store.as_retriever()
llm = ChatDatabricks(endpoint=llm_model, temperature=0.1, max_tokens=500)

SYSTEM_PROMPT = """
You are a helpful assistant. Use the following context to answer the question.

Response rules:
- Never create your own answer unless context is empty.
- Show the response in below format strictly:
  Answer: {{answer}}. Source: {{source}} 
- Must show the 'Answer' and 'Source' keywords and all must be in 1 line.
- If you cannot find the answer from the context, say "Answer: Cannot find the answer from the context. Source: Not applicable".
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context: {context}\n\nQuestion: {question}\n\nAnswer:")
])

def get_context(inputs):
    docs = retriever.invoke(inputs["question"])
    if not docs or all("Cannot find the answer from the context" in doc.page_content for doc in docs):
        return ""
    return "\n\n".join([f"Answer: {doc.page_content}  \n  Source: {doc.metadata.get('chunk_id', 'Unknown')}" for doc in docs])

rag_chain = RunnableSequence(
    {"context": get_context, "question": lambda x: x["question"]},
    prompt,
    llm
)

def get_internet_answer(question):
    # Use Databricks LLM to answer from public sources (no context)
    INTERNET_SYSTEM_PROMPT = """
    You are a helpful assistant. Content is from Internet or public sources. Show the response in below format: 
    Answer: **WARNING: Content is from Internet as answer cannot be found in context.** {{answer}}. Source: {{source}}.
    """
    internet_prompt = ChatPromptTemplate.from_messages([
        ("system", INTERNET_SYSTEM_PROMPT),
        ("human", f"Question: {question}\n\nAnswer:")
    ])
    return llm.invoke(internet_prompt.format_messages(question=question)).content

def invoke_rag_chain(question):
    query = {"question": question}
    response = rag_chain.invoke(query)
    if "Cannot find the answer from the context" in response.content or "Source: Not applicable" in response.content:
        internet_answer = get_internet_answer(question)
        return type("Response", (), {"content": internet_answer})()
    return response

st.title("RAG Q&A Application Demo")
question = st.text_input("Ask a question:")

if st.button("Submit") and question:   
    response = invoke_rag_chain(question)
    try:
        answer = response.content.strip().split("Answer:")[1].strip().split("Source:")[0].strip()
        source = response.content.strip().split("Source:")[1].strip()
    except:
        answer = "Please try again"
        source = "Unknown"
    st.markdown("### Answer")
    st.write(answer)
    st.markdown("### Sources")
    st.write(source)