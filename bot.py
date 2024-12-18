from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
from fastapi.responses import HTMLResponse

app = FastAPI()

class MessageRequest(BaseModel):
    content: str


# Read content from sample.txt and search for relevant text
def search_in_file(query: str, filename="/workspaces/hacky2/sample_text.txt"):
    with open(filename, "r") as file:
        content = file.read().lower()  # Read file content in lowercase for case-insensitive search
    
    # Simple search: look for the occurrence of the query (substring search)
    relevant_content = []
    for line in content.split("\n"):
        if query.lower() in line:
            relevant_content.append(line)
    
    return "\n".join(relevant_content) if relevant_content else "No relevant content found."


# API endpoint for calling the NVIDIA model
NVIDIA_API_KEY = "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"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


@app.post("/generate/")
async def generate_response(request: MessageRequest):
    # Search for relevant content in the file based on the user query
    relevant_content = search_in_file(request.content)
    
    # Combine the retrieved content with the user query
    prompt = f"User query: {request.content}\n\nRelevant content:\n{relevant_content}\n\nResponse:"
    #prompt = "answer the question only from the following data" + content + "the question is:" + request.content
    
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta/llama-3.1-70b-instruct",
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
                    speakResponse(data.response); // Speak the response aloud
                } else {
                    responseDiv.innerHTML = "<b>Error:</b> " + data.error;
                }
            }
            // Function to capture speech input
        function startSpeechRecognition() {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en-US';
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;

            recognition.start();

            recognition.onresult = function (event) {
                const transcript = event.results[0][0].transcript;
                document.getElementById("userInput").value = transcript;
            };

            recognition.onerror = function (event) {
                alert('Error occurred in recognition: ' + event.error);
            };
        }

        // Function to speak the bot's response
        function speakResponse(text) {
            const speechSynthesis = window.speechSynthesis;
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            speechSynthesis.speak(utterance);
        }
        </script>
    </head>
    <body>
        <h1>Welcome to the RAG Application</h1>
        <div>
            <label for="userInput">Enter your query:</label><br>
            <input type="text" id="userInput" name="userInput" size="50">
            <button onclick="startSpeechRecognition()">🎤 Mic</button>
        </div>
        <br>
        <button onclick="generateResponse()">Generate Response</button>
        <br><br>
        <div id="response"></div>
    </body>
    </html>
    """
    return html_content
