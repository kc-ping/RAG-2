import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
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

#combine the query and the relevant documents 
combined_input = f"""Based on the following context, answer the question: {query}
Documents:
{chr(10).join([doc.page_content for doc in relevant_docs])}
Please provide a concise answer based only on the information provided from these documents.
If the answer is not contained within the context, please respond with "I cannot find an answer to that question in the provided documents."""

#Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=500)
# temperature=0.2, max_tokens=500 because we want a concise answer

#Define th emessages for the model
messages = [
    SystemMessage(content="You are a helpful assistant that provides concise answers based on the provided context."),
    HumanMessage(content=combined_input)
]

#Invoke the model with the combined output
response = model.invoke(messages)

#Display the response
print("--- Model Response ---")
print(response.content)
