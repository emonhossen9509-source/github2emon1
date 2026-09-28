import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Hugging Face Inference API settings
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

STUDENT_NAME = os.environ.get("STUDENT_NAME", "Your Name Here")
STUDENT_ID = os.environ.get("STUDENT_ID", "Your Student ID Here")


@app.route("/")
def home():
    return render_template("index.html", name=STUDENT_NAME, student_id=STUDENT_ID)


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to summarize."}), 400

    if not HF_API_TOKEN:
        return jsonify({"error": "Server is missing HF_API_TOKEN. Please set it in environment variables."}), 500

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {"max_length": 130, "min_length": 30, "do_sample": False},
    }

    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Hugging Face model can return a list of dicts, or an error/loading message
        if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
            summary = result[0]["summary_text"]
            return jsonify({"summary": summary})
        elif isinstance(result, dict) and "error" in result:
            return jsonify({"error": result["error"]}), 503
        else:
            return jsonify({"error": "Unexpected response from AI model. Please try again."}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error contacting AI service: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
