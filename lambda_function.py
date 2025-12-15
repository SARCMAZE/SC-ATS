import json
import datetime
import boto3
from llm_service import call_llm

s3 = boto3.client("s3")
BUCKET_NAME = "ats-qualified-resumes-demo"

def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])

        resume_text = body.get("resume_text", "")
        job_role = body.get("job_role", "")

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
            key = f"{job_role}/resume_{datetime.datetime.now().isoformat()}.json"

            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=key,
                Body=json.dumps({
                    "job_role": job_role,
                    "ats_score": score,
                    "resume_text": resume_text,
                    "result": llm_output
                }),
                ContentType="application/json"
            )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"result": llm_output})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
