import os
import time
import requests
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from threading import Thread

# Load env vars
load_dotenv()

app = Flask(__name__)

# ENV vars
SLACK_VERIFICATION_TOKEN = os.getenv("SLACK_VERIFICATION_TOKEN")
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")

# Load model and Jira data
model = SentenceTransformer('all-MPNET-base-v2')
embeddings = joblib.load("jira_embeddings_v2.pkl")
df = joblib.load("jira_df_v2.pkl")

# Gemini setup
genai.configure(api_key=GEMINI_API_TOKEN)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def find_similar_tickets(query_text, top_n=5):
    query_vec = model.encode([query_text])
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results['Similarity'] = sims[top_indices] * 100
    return results

def analyze_issue_context(issue_key, summary, comments_dict):
    prompt = (
        f"You are a support assistant who has to summarize the cause of the issue based on the ticket details below. "
        f"Do not use proper nouns. Do not use emojis. Just give the cause of the ticket if possible. If not possible, just tell 'No cause/resolution found. Please create a ticket'. "
        f"Latest comments:\n{comments_dict['First']}\n\n"
        f"Largest comments:\n{comments_dict['Largest']}\n\n"
        f"Additional Comments:\n{comments_dict['Remaining']}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Failed to analyze this issue: {str(e)}"


@app.route('/similar', methods=['POST'])
def handle_similar():
    if request.form.get('token') != SLACK_VERIFICATION_TOKEN:
        return "Invalid verification token", 403

    query = request.form.get('text')
    response_url = request.form.get('response_url')

    def process_and_respond():
        results = find_similar_tickets(query)

        if results.empty or results['Similarity'].max() < 65:
            message = f"No similar tickets with confidence > 65% for: _{query}_\n👉 Please create a Jira ticket."
        else:
            lines = [f"*Top 5 Similar Issues for:* _{query}_\n"]
            for _, row in results.iterrows():
                comments_dict = {
                    "First": str(row['First 50% Comments (Most important context)'])[:500],
                    "Largest": str(row['Largest comments (Most probable resolutions and most context)'])[:500],
                    "Remaining": str(row['Remaining Comments'])[:500],
                }
                analysis = analyze_issue_context(row['Issue key'], row['Summary'], comments_dict)
                lines.append(
                    f"*<{JIRA_BASE_URL}/browse/{row['Issue key']}|{row['Issue key']}>* "
                    f"(`{row['Similarity']:.2f}%`)\n"
                    f"_{row['Summary']}_\n"
                    f"> {analysis}\n"
                )
            message = "\n".join(lines)

        # POST back to Slack
        try:
            requests.post(response_url, json={"response_type": "in_channel", "text": message})
        except Exception as e:
            print(f"❌ Slack response failed: {e}")

    # Respond immediately and process in background
    Thread(target=process_and_respond).start()
    return "", 200


@app.route('/create', methods=['POST'])
def create_ticket():
    if request.form.get('token') != SLACK_VERIFICATION_TOKEN:
        return "Invalid verification token", 403

    text = request.form.get('text')  # Expected format: "description || brand || environment"
    user = request.form.get('user_name')

    try:
        description_text, brand, environment = [x.strip() for x in text.split("||")]
    except ValueError:
        return jsonify(response_type="ephemeral", text="❌ Invalid format. Use:\n`/create description || brand || environment`")

    issue_data = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": f"Issue from Slack by {user}",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description_text}]
                    }
                ]
            },
            "issuetype": {"name": "Task"},
            "customfield_11997": [{"value": brand}],
            "customfield_11800": [{"value": environment}],
            "components": [{"name": "FT"}]
        }
    }

    response = requests.post(
        f"{JIRA_BASE_URL}/rest/api/3/issue",
        json=issue_data,
        auth=(JIRA_EMAIL, JIRA_API_TOKEN),
        headers={"Accept": "application/json", "Content-Type": "application/json"}
    )

    if response.status_code == 201:
        issue_key = response.json()["key"]
        return jsonify(response_type="in_channel", text=f"✅ Jira ticket created: {issue_key}")
    else:
        return jsonify(response_type="ephemeral", text=f"❌ Jira error: {response.text}")

if __name__ == "__main__":
    app.run(port=5000)
