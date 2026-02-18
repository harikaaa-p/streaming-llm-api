from flask import Flask, request, Response, stream_with_context
from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

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
            if chunk.choices[0].delta.content:
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


@app.route("/stream", methods=["POST"])
def stream():
    data = request.json

    if not data or "prompt" not in data:
        return {"error": "Prompt is required"}, 400

    prompt = data["prompt"]

    return Response(
        stream_with_context(generate_stream(prompt)),
        content_type="text/event-stream"
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
