from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
# Load environment variables
load_dotenv()
persistent_directory = "db/faiss_index"
# Connect to your document database
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
if not os.path.exists(os.path.join(persistent_directory, "index.faiss")):
    raise ValueError(
        f"The FAISS vector store does not exist at '{persistent_directory}'. Run ingestion_pipeline.py first."
    )
db = FAISS.load_local(persistent_directory, embedding_model, allow_dangerous_deserialization=True)

# Set up AI model
model = ChatOpenAI(model="gpt-4o")

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    document_context = "\n".join(f"- {doc.page_content}" for doc in docs)
    combined_input = (
        f"Based on the following documents, please answer this question: {user_question}\n\n"
        "Documents:\n"
        f"{document_context}\n\n"
        "Please provide a clear, helpful answer using only the information from these documents.\n"
        "If you can't find the answer in the documents, say \"I don't have enough information to answer that question based on the provided documents.\""
    )
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()