from flask import Flask, request, jsonify
import json
import datetime
from llm_service import call_llm

app = Flask(__name__)


from flask import send_from_directory

@app.route("/")
def serve_frontend():
    return send_from_directory("frontend", "index.html")

def store_in_cloud(resume, job_role, result):
    data = {
        "timestamp": str(datetime.datetime.now()),
        "job_role": job_role,
        "resume_text": resume,
        "ats_result": result
    }

    try:
        with open("cloud_storage.json", "r") as f:
            existing = json.load(f)
    except:
        existing = []

    existing.append(data)

    with open("cloud_storage.json", "w") as f:
        json.dump(existing, f, indent=4)

@app.route("/analyze", methods=["POST"])
def analyze():
    content = request.json

    resume_text = content.get("resume_text", "")
    job_role = content.get("job_role", "")

    prompt = f"""
You are an Applicant Tracking System (ATS).

Return output strictly in this format:
1. ATS Score (0–100)
2. Key Skills Detected
3. Missing Keywords
4. Formatting Issues
5. Final Decision (Accepted / Rejected)
6. Explanation (2–3 lines)
7. Disclaimer

Job Role: {job_role}

Resume Text:
{resume_text}
"""

    llm_output, score = call_llm(prompt)

    if score >= 70:
        store_in_cloud(resume_text, job_role, llm_output)

    return jsonify({"result": llm_output})

if __name__ == "__main__":
    app.run(debug=True)
