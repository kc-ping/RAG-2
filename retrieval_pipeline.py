import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/faiss_index"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.path.exists(os.path.join(persistent_directory, "index.faiss")):
    raise ValueError(
        f"The FAISS vector store does not exist at '{persistent_directory}'. Run ingestion_pipeline.py first."
    )

db = FAISS.load_local(persistent_directory, embedding_model, allow_dangerous_deserialization=True)

# Search for relevant documents
query = "What was NVIDIA's first graphics accelerator called?"

retriever = db.as_retriever(search_kwargs={"k": 5})
print(f"Retriever created with search_kwargs: {retriever.search_kwargs}")

relevant_docs = retriever.invoke(query)
print(f"Retrieved {len(relevant_docs)} relevant documents for the query.")

print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")