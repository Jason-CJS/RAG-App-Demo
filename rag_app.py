# Databricks notebook source
# DBTITLE 1,Install required libraries
# Enables the use of Databricks-native vector search APIs
%pip install databricks-vectorsearch==0.64

# For reading and parsing PDF files
%pip install pymupdf==1.26.7

# Adds Databricks-native LangChain integrations (e.g., ChatDatabricks, VectorSearchRetrieverTool)
%pip install databricks-langchain==0.14.0

%pip install "mlflow[databricks]==3.9.0"
%pip install pypdf==6.6.2
%pip install langchain-unstructured==1.0.1
%pip install unstructured==0.18.31
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Show all installed libraries
# MAGIC %skip
# MAGIC %pip list

# COMMAND ----------

# DBTITLE 1,Import required libraries
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings, DatabricksVectorSearch, VectorSearchRetrieverTool
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.errors.platform import ResourceAlreadyExists, ResourceDoesNotExist
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.service.vectorsearch import EndpointType
from databricks.vector_search.client import VectorSearchClient

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyMuPDFLoader #, UnstructuredFileLoader
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

from collections import defaultdict  # Group documents by source file
from pyspark.sql import SparkSession
from datetime import datetime
import uuid
import mlflow
import requests

# COMMAND ----------

# DBTITLE 1,Define all variables
# llm_model = "databricks-gpt-5-2"
# llm_model = "databricks-gpt-5-1"
# llm_model = "databricks-gpt-oss-20b"
# llm_model = "databricks-gpt-oss-120b"
# llm_model = "databricks-llama-4-maverick"
llm_model = "databricks-gemma-3-12b"

# embedding_model = "databricks-gte-large-en"
embedding_model = "databricks-bge-large-en"

vector_search_endpoint = "test_vector_search"

catalog_name = "workspace"
schema_name = "my_schema"
volume_name = "my_volume"
table_doc_text = f"{catalog_name}.{schema_name}.rag_doc_text"
table_doc_page = f"{catalog_name}.{schema_name}.rag_doc_page"
table_doc_chunk = f"{catalog_name}.{schema_name}.rag_doc_chunk"
table_doc_embedding = f"{catalog_name}.{schema_name}.rag_doc_embedding"

# COMMAND ----------

# DBTITLE 1,Drop all tables
spark.sql(f"DROP TABLE IF EXISTS {table_doc_text}")
spark.sql(f"DROP TABLE IF EXISTS {table_doc_page}")
spark.sql(f"DROP TABLE IF EXISTS {table_doc_chunk}")

# You must remove the vector search index before dropping the table. This ensures that all dependencies are cleaned up and Unity Catalog allows the operation. If you do not have permission to delete the index, contact your workspace administrator.

client = VectorSearchClient()
try:
    client.delete_index(endpoint_name=vector_search_endpoint, index_name=table_doc_embedding)
    spark.sql(f"DROP TABLE IF EXISTS {table_doc_embedding}")
except (ResourceDoesNotExist, NotFound):
    pass
except Exception as e:
    print(f"Error deleting index: {e}")

# COMMAND ----------

# DBTITLE 1,Helper functions
# Helper functions from https://docs.databricks.com/aws/en/notebooks/source/generative-ai/unstructured-data-pipeline.html

def volume_path() -> str:
    return f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"

def volume_uc_fqn() -> str:
    return f"{catalog_name}.{schema_name}.{volume_name}"
    
def check_if_volume_exists() -> bool:    
    w = WorkspaceClient()
    try:  
        w.volumes.read(name=volume_uc_fqn())
        return True
    except (ResourceDoesNotExist, NotFound):
        return False

def create_volume():
    try:
        w = WorkspaceClient()
        w.volumes.create(
            catalog_name=catalog_name,
            schema_name=schema_name,
            name=volume_name,
            volume_type=VolumeType.MANAGED,
        )
    except ResourceAlreadyExists:
        pass

def check_if_catalog_exists() -> bool:
    w = WorkspaceClient()
    try:
        w.catalogs.get(name=catalog_name)
        return True
    except (ResourceDoesNotExist, NotFound):
        return False

def check_if_schema_exists() -> bool:
    w = WorkspaceClient()
    try:
        full_name = f"{catalog_name}.{schema_name}"
        w.schemas.get(full_name=full_name)
        return True
    except (ResourceDoesNotExist, NotFound):
        return False

def create_or_validate_volume() -> tuple[bool, str]:
    """
    Validates that the volume exists and creates it if it doesn't
    Returns:
        tuple[bool, str]: A tuple containing (success, error_message).
        If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
    """    
    if not check_if_catalog_exists():
        msg = f"Catalog '{catalog_name}' does not exist. Please create it first."
        return (False, msg)

    if not check_if_schema_exists():
        msg = f"Schema '{schema_name}' does not exist in catalog '{catalog_name}'. Please create it first."
        return (False, msg)
    
    if not check_if_volume_exists():
        print(f"Volume '{volume_path()}' does not exist. Creating...")
        try:
            create_volume()
        except Exception as e:
            msg = f"Failed to create volume: {str(e)}"
            return (False, msg)
        msg = f"Successfully created volume '{volume_path()}'."
        print(msg)
        return (True, msg)

    msg = f"Volume '{volume_path()}' exists."
    print(msg)
    return (True, msg)

def list_files() -> list[str]:
    """
    Lists all files in the Unity Catalog volume using dbutils.fs.
    Returns:
        list[str]: A list of file paths in the volume
    Raises:
        Exception: If the volume doesn't exist or there's an error accessing it
    """
    if not check_if_volume_exists():
        raise Exception(f"Volume '{volume_path()}' does not exist")

    w = WorkspaceClient()
    try:
        # List contents using dbutils.fs
        files = w.dbutils.fs.ls(volume_path())
        return [file.name for file in files]
    except Exception as e:
        raise Exception(f"Failed to list files in volume: {str(e)}")

def check_if_vector_search_endpoint_exists():
    w = WorkspaceClient()
    vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
    if (sum([vector_search_endpoint == ve.name for ve in vector_search_endpoints])==0):
        return False
    else:
        return True

def create_vector_search_endpoint():
    w = WorkspaceClient()
    print(f"Please wait, creating Vector Search endpoint '{vector_search_endpoint}'. This can take up to 20 minutes...")
    w.vector_search_endpoints.create_endpoint_and_wait(vector_search_endpoint, endpoint_type=EndpointType.STANDARD)
    # Make sure vector search endpoint is online and ready.
    w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(vector_search_endpoint)

def validate_vector_search_endpoint() -> tuple[bool, str]:
    """
    Validates that the specified Vector Search endpoint exists
    Returns:
        tuple[bool, str]: A tuple containing (success, error_message).
        If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
    """
    if not check_if_vector_search_endpoint_exists():
        msg = f"Vector Search endpoint '{vector_search_endpoint}' does not exist. Please call 'create_or_validate_vector_search_endpoint()' to create it."
        return (False, msg)

    msg = f"Vector Search endpoint '{vector_search_endpoint}' exists."
    return (True, msg)

def create_or_validate_vector_search_endpoint():
    if not check_if_vector_search_endpoint_exists():
        create_vector_search_endpoint()
    return validate_vector_search_endpoint()

# COMMAND ----------

# DBTITLE 1,Create or check Volume and Vector Search endpoint
create_or_validate_volume()
create_or_validate_vector_search_endpoint()

# COMMAND ----------

# DBTITLE 1,Download files from Internet and save in Volume
download_urls = [
"https://www.uwa.edu.au/about/-/media/project/uwa/uwa/about/docs/campus-management/facilities-management/a-uwa-design-and-construction-standards-building-and-architecture.pdf",
"https://www.uwa.edu.au/life-at-uwa/-/media/project/uwa/uwa/lifeatuwa/docs/accommodation/2025-albany-accommodation--living-guide.pdf",
"https://www.uwa.edu.au/students/-/media/project/uwa/uwa/students/docs/important-dates-2025.pdf",
"https://www.the-guild.eu/publications/position-papers/the-guild-s-position-paper-on-the-use-of-ai-in-research_nov2024.pdf"
]

for url in download_urls:
    file_name = url.split("/")[-1]
    file_path = f"{volume_path()}{file_name}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")

# COMMAND ----------

# DBTITLE 1,Load all defined files based on file types
file_types = ["pdf", "docx", "pptx", "xlsx", "txt"]

documents = []
for ext in file_types:
    files = dbutils.fs.ls(volume_path())
    for file in files:
        if file.name.lower().endswith(f".{ext}"):
            file_path = f"{volume_path()}{file.name}"
            if ext == "pdf":
                loader = PyPDFDirectoryLoader(volume_path())
                documents.extend(loader.load())
                break  # PyPDFDirectoryLoader loads all PDFs in the directory
            else:
                # loader = UnstructuredFileLoader(file_path)
                loader = UnstructuredLoader(file_path)
                documents.extend(loader.load())

# COMMAND ----------

# DBTITLE 1,Store all document text for each file
spark = SparkSession.builder.getOrCreate()

system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

grouped_docs = defaultdict(list)
for doc in documents:
    source = doc.metadata.get("source", None)
    grouped_docs[source].append(doc)

text = []
for source, docs in grouped_docs.items():
    doc_id = str(uuid.uuid4())
    text.append({
        "doc_id": doc_id,
        "doc_text": " ".join([doc.page_content for doc in docs]),        
        "file_type": source.split('.')[-1] if source and '.' in source else None,
        "modified_at": docs[0].metadata.get("moddate", None) if docs else None,
        "source": source,
        "total_page": len(docs) if docs else None,
        "uploaded_at": system_time
    })

df = spark.createDataFrame(text)
df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table_doc_text)

# COMMAND ----------

# DBTITLE 1,Store document text for each page
system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

pages = []
for item in text:
    doc_id = item["doc_id"]
    source = item["source"]    
    docs = grouped_docs.get(source, [])
    for document in docs:
        pages.append({
            "doc_id": doc_id,
            "source": source,            
            "page_number": document.metadata.get("page", None) if document else None,
            "page_text": document.page_content,
            "uploaded_at": system_time
        })

df = spark.createDataFrame(pages)
df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table_doc_page)

# COMMAND ----------

# DBTITLE 1,Chunk the document text for each page without chunk_vector
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " "])
system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

chunks = []
for item in text:
    doc_id = item["doc_id"]
    source = item["source"]
    docs = grouped_docs.get(source, [])    
    docs_new = docs[2:] if source.split("/")[-1].split(".")[-1].lower() == "pdf" else docs  # Exclude cover (page 0) and ToC (page 1) for pdf files
    split_chunks = text_splitter.split_documents(docs_new)
    for idx, chunk in enumerate(split_chunks):
        # chunk_id = f"{chunk.metadata.get('source', None).split('/')[-1] if chunk and chunk.metadata.get('source', None) else None}_{str(uuid.uuid4())}"
        chunk_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        chunk_id = f"{chunk.metadata.get('source', None).split('/')[-1] if chunk and chunk.metadata.get('source', None) else None}_{chunk_time}"
        # chunk_vector = embedding_model_instance.embed_documents([chunk.page_content])[0]
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_text": chunk.page_content,
            "chunk_order": idx,
            # "chunk_vector": chunk_vector, 
            "page_number": chunk.metadata.get("page", None) if chunk else None,
            "uploaded_at": system_time
        })

# print(chunks)
df = spark.createDataFrame(chunks)
df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table_doc_chunk)


# COMMAND ----------

# DBTITLE 1,Chunk the document text for each page with chunk_vector
# MAGIC %skip
# MAGIC text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " "])
# MAGIC system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# MAGIC embedding_model_instance = DatabricksEmbeddings(endpoint=embedding_model)
# MAGIC
# MAGIC chunks = []
# MAGIC for item in text:
# MAGIC     doc_id = item["doc_id"]
# MAGIC     source = item["source"]
# MAGIC     docs = grouped_docs.get(source, [])    
# MAGIC     docs_new = docs[2:] if source.split("/")[-1].split(".")[-1].lower() == "pdf" else docs  # Exclude cover (page 0) and ToC (page 1) for pdf files
# MAGIC     split_chunks = text_splitter.split_documents(docs_new)
# MAGIC     for idx, chunk in enumerate(split_chunks):
# MAGIC         # chunk_id = f"{chunk.metadata.get('source', None).split('/')[-1] if chunk and chunk.metadata.get('source', None) else None}_{str(uuid.uuid4())}"
# MAGIC         chunk_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")
# MAGIC         chunk_id = f"{chunk.metadata.get('source', None).split('/')[-1] if chunk and chunk.metadata.get('source', None) else None}_{chunk_time}"
# MAGIC         chunk_vector = embedding_model_instance.embed_documents([chunk.page_content])[0]
# MAGIC         chunks.append({
# MAGIC             "doc_id": doc_id,
# MAGIC             "chunk_id": chunk_id,
# MAGIC             "chunk_text": chunk.page_content,
# MAGIC             "chunk_order": idx,
# MAGIC             "chunk_vector": chunk_vector,
# MAGIC             "page_number": chunk.metadata.get("page", None) if chunk else None,
# MAGIC             "uploaded_at": system_time
# MAGIC         })
# MAGIC
# MAGIC # print(chunks)
# MAGIC df = spark.createDataFrame(chunks)
# MAGIC df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table_doc_chunk)

# COMMAND ----------

# DBTITLE 1,Alter the chunk table properties
spark.sql(f"ALTER TABLE {table_doc_chunk} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# DBTITLE 1,Create and index embedding table with auto creation of embedding vector column
# Embedding vector refers to the actual numerical representation (usually a list or array of floats) that encodes information about an object (such as a word, sentence, image, or document) in a high-dimensional space. For example, in NLP, a word embedding vector represents a word's meaning.

# Create and index embedding table with auto creation of embedding vector column (_db_chunk_text_vector)

try:
    client = VectorSearchClient()
    vector_search_index = client.create_delta_sync_index(
        endpoint_name=vector_search_endpoint,
        index_name=table_doc_embedding,
        source_table_name=table_doc_chunk,
        pipeline_type="TRIGGERED",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=embedding_model,
        primary_key="chunk_id",
        # embedding_vector_column="chunk_vector",
        # embedding_dimension=1024,
    )
except Exception as e:
    print(e)
    pass

# COMMAND ----------

# DBTITLE 1,Test the RAG
# Set up MLflow tracking to Databricks
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/RAG-App-Demo")

SYSTEM_PROMPT = """
You are a helpful assistant. Use the following context to answer the question.

Response rules:
- Never create your own answer unless context is empty.
- Show the response in below format strictly:
  Answer: {{answer}}. Source: {{source}} 
- Must show the 'Answer' and 'Source' keywords and all must be in 1 line.
- If you cannot find the answer from the context, say "Answer: Cannot find the answer from the context. Source: Not applicable".
"""

vector_store = DatabricksVectorSearch(
    endpoint=vector_search_endpoint,
    index_name=table_doc_embedding,
    # If using chunk_vector column manually, then enable embedding and text_column parameters.
    # The index 'workspace.my_schema.rag_doc_embedding' uses Databricks-managed embeddings. Do not pass the `embedding` parameter when initializing vector store. 
    # embedding=DatabricksEmbeddings(endpoint=embedding_model),
    # The index 'workspace.my_schema.rag_doc_embedding' has the source column configured as 'chunk_text'. Do not pass the `text_column` parameter.
    # text_column="chunk_text"
)

retriever = vector_store.as_retriever()
llm = ChatDatabricks(endpoint=llm_model, temperature=0.1, max_tokens=500)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context: {context}\n\nQuestion: {question}\n\nAnswer:")
])

@mlflow.trace
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

@mlflow.trace
def get_internet_answer(question):
    # Use Databricks LLM to answer from public sources (contextless)
    INTERNET_SYSTEM_PROMPT = """
    You are a helpful assistant. Content is from Internet or public sources. Show the response in below format: 
    Answer: **WARNING: Content is from Internet as answer cannot be found in context.**\n{{answer}}. Source: {{source}}.
    """
    internet_prompt = ChatPromptTemplate.from_messages([
        ("system", INTERNET_SYSTEM_PROMPT),
        ("human", f"Question: {question}\n\nAnswer:")
    ])
    return llm.invoke(internet_prompt.format_messages(question=question)).content

@mlflow.trace
def invoke_rag_chain(inputs):
    response = rag_chain.invoke(inputs)
    if "Cannot find the answer from the context" in response.content or "Source: Not applicable" in response.content:
        return get_internet_answer(inputs["question"])
    return response.content

query = {"question": "Tell me about the UWA timetable 2025"}
# query = {"question": "representation of european language data"}
# query = {"question": "Sea lion"}

run_name = f"RAG on {datetime.now().strftime('%Y-%m-%d')}"
with mlflow.start_run(run_name=run_name) as run:
    response = invoke_rag_chain(query)
    try:
        answer = response.strip().split("Answer:")[1].strip().split("Source:")[0].strip()
        source = response.strip().split("Source:")[1].strip()
    except:
        answer = "Please try again"
        source = "Unknown"
    print(f"answer={answer}")
    print(f"source={source}")
    mlflow.log_param("question", query["question"])
    mlflow.log_param("answer", answer)
    mlflow.log_param("source", source)
    mlflow.log_param("llm_model", llm_model)
    mlflow.log_param("embedding_model", embedding_model)
    mlflow.log_param("vector_search_endpoint", vector_search_endpoint)
    mlflow.log_param("index_name", table_doc_embedding)

    mlflow.log_metric("response_length", len(response))
    mlflow.log_text(str(response), "response.txt")
    mlflow.log_dict({"content": response}, "response_full.json")
    print(response)
    print(f"Experiment run_id: {run.info.run_id}")
