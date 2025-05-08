import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
import time
import google.generativeai as genai
import os
from dotenv import load_dotenv


# Load model and data
model = SentenceTransformer('all-MPNET-base-v2')
embeddings = joblib.load("jira_embeddings_v2.pkl")
df = joblib.load("jira_df_v2.pkl")

# Similarity search with timing logs
def find_similar_tickets(query_text, top_n=5):
    start_time = time.time()
    t0 = time.time()
    query_vec = model.encode([query_text])
    t1 = time.time()
    sims = cosine_similarity(query_vec, embeddings)[0]
    t2 = time.time()

    top_indices = np.argsort(sims)[::-1][:top_n]
    results = df.iloc[top_indices][['Issue key', 'Summary',
                                    'First 50% Comments (Most important context)',
                                    'Largest comments (Most probable resolutions and most context)',
                                    'Remaining Comments']].copy()
    results['Similarity'] = sims[top_indices] * 100

    print(f" Query encoding time:       {t1 - t0:.4f}s")
    print(f"Cosine similarity time:    {t2 - t1:.4f}s")
    

    return results
def analyze_issue_context(issue_key, summary, comments_dict):
    genai.configure(api_key=os.getenv("GEMINI_API_TOKEN"))

    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    start_prompt = time.time()
    prompt = (
        f"You are a support assistant who has to summarize the cause of the issue based on the ticket details below. "
        f"Do not use proper nouns. Do not use emojis. Just give the cause of the ticket if possible. If not possible, just tell 'No cause/resolution found. Please create a ticket'. "
        f"Based on the details below, provide a concise cause/resolution (max 200 characters):\n\n"
        f"Latest comments on the Jira ticket which might have the cause of the issue:\n{comments_dict['First']}\n\n"
        f"Largest comments on the Jira ticket which might have the cause/context:\n{comments_dict['Largest']}\n\n"
        f"Additional Comments:\n{comments_dict['Remaining']}\n\n"
        f"Based on this, identify the likely cause of this issue. Keep the answer under 400 characters total."
    )

    try:
        response = model.generate_content(prompt)
        end_prompt = time.time()
        print(f"Prompt time:         {end_prompt - start_prompt:.4f}s\n")
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Failed to analyze this issue: {str(e)}"
# Main execution
if __name__ == "__main__":
    query = "Partner program was not allocated to the user. User made transaction to link the partner program but did not get linked. The user has a partner program already and the date of ending still shows same, even after making the linking transaction. Please fix on priority"
    start_main = time.time()
    results = find_similar_tickets(query)
    end_main = time.time()
    if results.empty or results['Similarity'].max() < 65:
        message = f"No similar tickets found with confidence > 65% for: _{query}_\n👉 Please create a new Jira ticket."
    else:
        lines = [f"*Top 5 Similar Issues for:* _{query}_\n"]
        for _, row in results.iterrows():
            comments_dict = {
                "First": str(row['First 50% Comments (Most important context)'])[:500],
                "Largest": str(row['Largest comments (Most probable resolutions and most context)'])[:500],
                "Remaining": str(row['Remaining Comments'])[:500],
            }
            analysis = analyze_issue_context(row['Issue key'], row['Summary'], comments_dict)

            line = (
                f"*<{ 'https://capillarytech.atlassian.net/browse/' + row['Issue key'] }|{row['Issue key']}>* "
                f"(`{row['Similarity']:.2f}%`)\n"
                f"_{row['Summary']}_\n"
                f"> {analysis}\n"
            )
            lines.append(line)

        message = "\n".join(lines)

    # Send to Slack
    slack_url = "https://v1.nocodeapi.com/harikishan/slack/IgAThybWcFBAkNHJ/sendText"
    slack_params = {"text": message}

    try:
        response = requests.post(url=slack_url, params=slack_params)
        response.raise_for_status()
        print("✅ Message sent to Slack.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Slack error: {e}")

    end_main = time.time()
    print(f"Total search time:         {end_main - start_main:.4f}s\n")

