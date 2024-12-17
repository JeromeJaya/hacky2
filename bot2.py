from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.responses import HTMLResponse

app = FastAPI()

class MessageRequest(BaseModel):
    content: str


# Function to read and process the file
def search_in_file(query: str, filename="sample.txt"):
    # Read the content from sample.txt
    with open(filename, "r") as file:
        content = file.read().splitlines()  # Split into lines for easier processing
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Combine the query and content to create a document set
    documents = [query] + content

    # Transform the documents into TF-IDF features
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarities between the query and the content
    cosine_similarities = np.array(tfidf_matrix[0].dot(tfidf_matrix.T).todense()).flatten()

    # Get the index of the most similar line(s)
    most_similar_index = cosine_similarities[1:].argmax()  # Skip the query itself (index 0)

    # Return the most relevant content
    return content[most_similar_index] if cosine_similarities[1:].max() > 0 else "No relevant content found."


# API endpoint for calling the NVIDIA model
NVIDIA_API_KEY = "your-nvidia-api-key"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


@app.post("/generate/")
async def generate_response(request: MessageRequest):
    # Search for relevant content in the file based on the user query
    relevant_content = search_in_file(request.content)
    
    # Combine the retrieved content with the user query
    prompt = f"User query: {request.content}\n\nRelevant content:\n{relevant_content}\n\nResponse:"
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta/llama-3.1-70b-instruct",  # Replace with your model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": False
    }

    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Check for errors
        response_json = response.json()
        return {"response": response_json["choices"][0]["message"]["content"]}

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# Route to serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def get_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Application</title>
        <script>
            async function generateResponse() {
                const userInput = document.getElementById("userInput").value;
                const responseDiv = document.getElementById("response");

                const response = await fetch('/generate/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ content: userInput })
                });

                const data = await response.json();

                if (data.response) {
                    responseDiv.innerHTML = "<b>Response:</b> " + data.response;
                } else {
                    responseDiv.innerHTML = "<b>Error:</b> " + data.error;
                }
            }
        </script>
    </head>
    <body>
        <h1>Welcome to the RAG Application</h1>
        <div>
            <label for="userInput">Enter your query:</label><br>
            <input type="text" id="userInput" name="userInput" size="50">
        </div>
        <br>
        <button onclick="generateResponse()">Generate Response</button>
        <br><br>
        <div id="response"></div>
    </body>
    </html>
    """
    return html_content
