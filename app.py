import os
from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient

app = Flask(__name__)

# ==========================================
# Student Information
# ==========================================

STUDENT_NAME = os.environ.get(
    "STUDENT_NAME",
    "Your Name Here"
)

STUDENT_ID = os.environ.get(
    "STUDENT_ID",
    "Your Student ID Here"
)


# ==========================================
# Hugging Face API
# ==========================================

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

if not HF_API_TOKEN:
    print("WARNING: HF_API_TOKEN is not configured.")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_TOKEN
)


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        name=STUDENT_NAME,
        student_id=STUDENT_ID
    )


# ==========================================
# Summarization
# ==========================================

@app.route("/summarize", methods=["POST"])
def summarize():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Invalid request."
        }), 400

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter some text to summarize."
        }), 400

    if not HF_API_TOKEN:
        return jsonify({
            "error": "HF_API_TOKEN is missing."
        }), 500

    try:

        result = client.summarization(
            text,
            model="facebook/bart-large-cnn"
        )

        # Hugging Face returns generated_text
        summary = result.summary_text

        return jsonify({
            "summary": summary
        })

    except Exception as e:

        print("Hugging Face Error:", str(e))

        return jsonify({
            "error": f"Error contacting AI service: {str(e)}"
        }), 500


# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )