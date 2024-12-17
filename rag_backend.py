import os
from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings.nvidia import NVIDIAEmbeddings
from langchain.chains import RetrievalQA
from langchain_nvidia_ai_endpoints import NVIDIAAIModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Flask App
app = Flask(__name__)

# Step 1: Configure API Key
os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"

# Step 2: Load NVIDIA Embeddings
embeddings = NVIDIAEmbeddings()

# Step 3: Load Documents (Your Text Corpus)
loader = TextLoader("sample_text.txt")  # Input file with knowledge base
documents = loader.load()

# Step 4: Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Step 5: Create a Vector Store for Retrieval
vector_store = FAISS.from_documents(texts, embeddings)

# Step 6: Load NVIDIA NIM Model for Generation
model = NVIDIAAIModel(model="chat-nvidia", api_key=os.environ["NVIDIA_API_KEY"])

# Step 7: Create RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# Step 8: Flask API Endpoint
@app.route("/query", methods=["POST"])
def query_rag():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Process Query with RAG
    result = qa_chain.run(user_query)
    return jsonify({"response": result})

# Run Flask Server
if __name__ == "__main__":
    app.run(debug=True)
