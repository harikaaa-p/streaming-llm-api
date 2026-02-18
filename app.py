from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from groq import Groq
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Enable full CORS support
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
)

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_stream(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content

                data = {
                    "choices": [
                        {
                            "delta": {
                                "content": content
                            }
                        }
                    ]
                }

                yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


@app.route("/stream", methods=["POST", "OPTIONS"])
def stream():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.json

    if not data or "prompt" not in data:
        return jsonify({"error": "Prompt is required"}), 400

    prompt = data["prompt"]

    return Response(
        stream_with_context(generate_stream(prompt)),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


# Proper Render port binding
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
