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

import time

from getComponents import get_jira_components , get_jira_environments, match_multiple_items
# Load env vars
load_dotenv()

app = Flask(__name__)

# ENV vars
SLACK_VERIFICATION_TOKEN = os.getenv("SLACK_VERIFICATION_TOKEN")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")

PRIORITY_SHEET = os.getenv("PRIORITY_SHEET")

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
- Be careful with numbers - For example "more than 150" is the same as "greater than 100", "less than 9" is the same as "less than 10", etc...(for example)
- More than = greater than, below = less than, etc...
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


@app.route('/pixie-similar', methods=['POST'])
def handle_similar():
    print(repr(JIRA_API_TOKEN))
    start_similar_time = time.time()
    if request.form.get('token') != SLACK_VERIFICATION_TOKEN:
        return "Invalid verification token", 403

    query = request.form.get('text')
    response_url = request.form.get('response_url')

    def process_and_respond():
        results = find_similar_tickets(query)
        high_conf_results = results[results['Similarity'] > 50]
        
        if high_conf_results.empty:
            predicted_priority = predict_priority_with_gemini(query)
            message = f"Predicted priority based on issue {predicted_priority}\n\nPixie✨ is not confident in any resolution for this query :(  _{query}_\nUse `/pixie-create` to create a new ticket."
            
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
                f"Pixie✨ found something!\n*Predicted Priority:* *{predicted_priority}*\nYou can check priorities *<{PRIORITY_SHEET}|here>*\n_______________________\n"
                f"*<{JIRA_BASE_URL}/browse/{top_row['Issue key']}|{top_row['Issue key']}>* "
                f"(`{top_row['Similarity']:.2f}%`)\n"
                f"_{top_row['Summary']}_\n"
                f"> {analysis}\n"
                
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
            lines.append("If you are not satisfied with Pixie's✨ resolution, she can help you crate a new JIRA ticket : `/pixie-create description | brand | environment | component | issue_type(Bug/Task)`")

            message = "\n".join(lines)


        # POST back to Slack
        try:
            requests.post(response_url, json={"response_type": "in_channel", "text": message})
            end_similar_time=time.time()
            exec_time = end_similar_time-start_similar_time
            print(f"Execution time : {exec_time:.4f} seconds")
        except Exception as e:
            print(f"❌ Slack response failed: {e}")

    # Respond immediately and process in background
    Thread(target=process_and_respond).start()
    return jsonify(
    response_type="ephemeral",
    text="Pixie✨ is working really hard to give you the best response, give her a minute..."
    ), 200





@app.route('/pixie-create', methods=['POST'])
def create_ticket():
    if request.form.get('token') != SLACK_VERIFICATION_TOKEN:
        return "Invalid verification token", 403

    text = request.form.get('text')  # Expected format: "description | brand | environment | component | issue_type(Bug/Task)"
    user = request.form.get('user_name')

    try:
        description_text, brand, environment, component, issue_type = [x.strip() for x in text.split("|")]
    except ValueError:
        return jsonify(response_type="ephemeral", text="Pixie✨ doesn't understand this syntax :( \nUse: `/pixie-create description | brand | environment | component | issue_type(Bug/Task)`\n")

    valid_components = get_jira_components()
    # valid_environments = get_jira_environments()

    # Split user input (assumes comma-separated)
    env_inputs = [e.strip() for e in environment.split(",")]
    comp_inputs = [c.strip() for c in component.split(",")]

    # Match
    # matched_envs = match_multiple_items(env_inputs, valid_environments)
    matched_comps = match_multiple_items(comp_inputs, valid_components)
    # print(f"env : {matched_envs}")
    print(f"comp : {matched_comps}")


    # Create issue payload
    issue_data = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": f"{description_text} : Pixie✨",
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
            "issuetype": {"name": issue_type},
            "customfield_11997": [{"value": brand}],
            "customfield_11800": [{"value": env} for env in env_inputs],
            "components": [{"name": comp} for comp in matched_comps]
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

        # ✨ Find and post similar tickets
        try:
            similar_results = find_similar_tickets(description_text)
            high_conf = similar_results[similar_results['Similarity'] > 50]

            if not high_conf.empty:
                comment_lines = ["Pixie✨ found some similar JIRA tickets that may help you:\n"]
                for _, row in high_conf.head(3).iterrows():
                    issue_url = f"{JIRA_BASE_URL}/browse/{row['Issue key']}"
                    comment_lines.append(
                        f"- {row['Issue key']} : {issue_url}   "
                        f"({row['Similarity']:.1f}% match) : {row['Summary']}"
                    )
                comment_body = "\n\n".join(comment_lines)

                comment_payload = {
  "body": {
    "type": "doc",
    "version": 1,
    "content": [
      {
        "type": "paragraph",
        "content": [
          {
            "type": "text",
            "text": comment_body
          }
        ]
      }
    ]
  }
}

                requests.post(
                    f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/comment",
                    json=comment_payload,
                    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
                    headers={"Accept": "application/json", "Content-Type": "application/json"}
                )

        except Exception as e:
            print(f"⚠️ Failed to add similar ticket comment: {e}")

        return jsonify(
            response_type="in_channel",
            text=f"✅ Pixie✨ created a new ticket: *<{JIRA_BASE_URL}/browse/{issue_key}|{issue_key}>*"
        )

    else:
        return jsonify(response_type="ephemeral", text=f"❌ Pixie✨ received this error from JIRA: {response.text}")



if __name__ == "__main__":
    app.run(port=5000)
