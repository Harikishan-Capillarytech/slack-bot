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
# priority_df = joblib.load("priority_rules_df.pkl")
# priority_embeddings = joblib.load("priority_rules_embeddings.pkl")

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

rules_df = pd.read_csv("priority_rules.csv")
rules_df.columns = rules_df.columns.str.strip()

def get_structured_rules(df):
    rules = []
    for _, row in df.iterrows():
        rule = {
            "Feature/Product": str(row.get("Feature/Product", "")).strip(),
            "Use Case": str(row.get("Use Case", "")).strip(),
            "Component": str(row.get("Jira Component", "")).strip(),
            "P0": str(row.get("P0", "")).strip(),
            "P1": str(row.get("P1", "")).strip(),
            "P2": str(row.get("P2", "")).strip(),
            "P3": str(row.get("P3", "")).strip(),
        }
        rules.append(rule)
    return rules

structured_rules = get_structured_rules(rules_df)


def predict_priority_with_gemini(issue_description):
    prompt = f"""
You are a Jira priority classification assistant.

Given the description of a Jira issue and a set of prioritization rules in table format, respond with the *most appropriate predicted priority* (one of: P0, P1, P2, P3). Also mention the matching component and product if relevant.

Issue:
\"\"\"
{issue_description}
\"\"\"

Rules:
{structured_rules}

Instructions:
- Match based on component, product, or use-case.
- Respect hard constraints (e.g., "Import issues cannot be P1 or P0").
- If no strong match is found, default to P3.

Respond only with:
Predicted Priority: P1 (for example)
Matching Component: XYZ
Matching Product: ABC
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def analyze_issue_context(issue_key, summary, comments_dict):
    prompt = (
        f"You are a support assistant who has to summarize the cause of the issue based on the ticket details below. "
        f"Do not use proper nouns. Do not use emojis. Just give the cause of the ticket if possible. If not possible, just tell 'No cause/resolution found. Please check the comments on this ticket to get a better understanding"
        f"Latest comments:\n{comments_dict['First']}\n\n"
        f"Largest comments:\n{comments_dict['Largest']}\n\n"
        f"Additional Comments:\n{comments_dict['Remaining']}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        tokens_used = 0
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            tokens_used = response.prompt_feedback.total_tokens

        print(f"🔢 Gemini API used {tokens_used} tokens for this query.")
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
        high_conf_results = results[results['Similarity'] > 50]
        
        if high_conf_results.empty:
            message = f"No similar tickets with high confidence for: _{query}_\nUse `/create` to create a new ticket."
        else:
            lines = [f"*Top Similar Issues for:* _{query}_\n"]

            # Run priority prediction only on the first (most confident) result
            top_row = high_conf_results.iloc[0]
            comments_dict = {
                "First": str(top_row['First 50% Comments (Most important context)'])[:500],
                "Largest": str(top_row['Largest comments (Most probable resolutions and most context)'])[:500],
                "Remaining": str(top_row['Remaining Comments'])[:500],
            }
            analysis = analyze_issue_context(top_row['Issue key'], top_row['Summary'], comments_dict)
            issue_description = top_row['Summary'] + " " + top_row['Largest comments (Most probable resolutions and most context)']
            predicted_priority = predict_priority_with_gemini(issue_description)

            lines.append(
                f"*<{JIRA_BASE_URL}/browse/{top_row['Issue key']}|{top_row['Issue key']}>* "
                f"(`{top_row['Similarity']:.2f}%`)\n"
                f"_{top_row['Summary']}_\n"
                # f"> {analysis}\n"
                f"*Predicted Priority:* {predicted_priority}\n"
            )

            # Add the rest of the high-confidence results (skip the first)
            for _, row in high_conf_results.iloc[1:].iterrows():
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
    return jsonify(
    response_type="ephemeral",
    text="Loading similar tickets... Please wait a few seconds."
    ), 200

def predict_priority(issue_text):
    issue_embedding = model.encode([issue_text], normalize_embeddings=True)
    sims = cosine_similarity(issue_embedding, priority_embeddings)[0]
    best_idx = np.argmax(sims)

    if sims[best_idx] < 0.40:
        return "No matching rule found"

    row = priority_df.iloc[best_idx]
    for level in [' P0', 'P1', 'P2', 'P3']:
        if str(row[level]).strip().lower() == 'yes':
            return f"{row['Feature/Product']} | {row['Component']} → Qualifies as ≤{level}"

    return f"{row['Feature/Product']} | {row['Component']} → Only P3"



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
        return jsonify(
    response_type="in_channel",
    text=f"✅ Jira ticket created: *<{JIRA_BASE_URL}/browse/{issue_key}|{issue_key}>*")
    else:
        return jsonify(response_type="ephemeral", text=f"❌ Jira error: {response.text}")
    



if __name__ == "__main__":
    app.run(port=5000)
