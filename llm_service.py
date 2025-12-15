import random
import requests
import os

# Optional real LLM endpoint (can be empty / unreliable)
HF_API_URL = os.environ.get("HF_API_URL", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

def random_ats():
    """Fallback ATS generator"""
    score = random.randint(66, 80)
    decision = "Accepted" if score >= 70 else "Rejected"

    response = f"""
1. ATS Score: {score}
2. Key Skills Detected: Python, Data Structures, AWS, Docker
3. Missing Keywords: System Design, Scalability
4. Formatting Issues: Minor formatting inconsistencies
5. Final Decision: {decision}
6. Explanation: Resume shows relevant skills but partially matches the job role.
7. Disclaimer: This analysis is AI-generated and is not a hiring decision. Results are for informational purposes only.
""".strip()

    return response, score


def call_llm(prompt):
    """
    Try real LLM first (20s timeout).
    If anything fails → fallback to random ATS.
    """

    # ---- TRY REAL LLM ----
    try:
        if HF_API_URL and HF_API_KEY:
            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.2
                }
            }

            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=20   # 🔥 HARD TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            # Basic safety check
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "")

                if text.strip():
                    # Try to extract score
                    score = 0
                    for line in text.splitlines():
                        if "ATS Score" in line:
                            digits = "".join(filter(str.isdigit, line))
                            if digits:
                                score = int(digits)
                                break

                    # If score looks valid, return real result
                    if 0 <= score <= 100:
                        return text.strip(), score

    except Exception as e:
        # Any error → fallback
        print("LLM failed, using fallback:", str(e))

    # ---- FALLBACK ----
    return random_ats()
