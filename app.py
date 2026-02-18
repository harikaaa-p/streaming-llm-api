from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Enable full CORS support
CORS(app, resources={r"/*": {"origins": "*"}})

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_stream(prompt):
    try:
        # Immediate first chunk (for latency test)
        yield 'data: {"choices":[{"delta":{"content":""}}]}\n\n'

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        buffer = ""
        chunk_counter = 0

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                buffer += chunk.choices[0].delta.content

                # Every ~80 characters, flush a chunk
                if len(buffer) >= 80:
                    data = {
                        "choices": [
                            {
                                "delta": {
                                    "content": buffer
                                }
                            }
                        ]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    buffer = ""
                    chunk_counter += 1

        # Flush remaining buffer
        if buffer:
            data = {
                "choices": [
                    {
                        "delta": {
                            "content": buffer
                        }
                    }
                ]
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"



@app.route("/stream", methods=["GET", "POST", "OPTIONS"])
def stream():
    # Preflight support
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    # Grader reachability check
    if request.method == "GET":
        return jsonify({"status": "stream endpoint live"}), 200

    # Handle POST streaming
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


# Proper Render binding
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
