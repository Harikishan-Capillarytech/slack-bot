import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
import time
import pandas as pd

# Load saved data
embeddings = joblib.load("jira_embeddings_v2.pkl")
df = joblib.load("jira_df_v2.pkl")

# Load same model
model = SentenceTransformer('all-MPNET-base-v2')

# Similarity search function with timing
def find_similar_tickets(query_text, top_n=5):
    start_time = time.time()

    # Embed query
    t0 = time.time()
    query_vec = model.encode([query_text])
    t1 = time.time()

    # Cosine similarity
    sims = cosine_similarity(query_vec, embeddings)[0]
    t2 = time.time()

    # Get top matches
    top_indices = np.argsort(sims)[::-1][:top_n]
    results = df.iloc[top_indices][['Issue key', 'Summary', 'First 50% Comments (Most important context)']].copy()
    results['Similarity'] = sims[top_indices] * 100

    end_time = time.time()

    # Print timing details
    print(f"🧠 Query encoding time:       {t1 - t0:.4f}s")
    print(f"📈 Cosine similarity time:    {t2 - t1:.4f}s")
    print(f"⏱️ Total search time:         {end_time - start_time:.4f}s\n")

    return results


# Format for Slack
def format_results_for_slack(results):
    lines = ["*Top Similar Jira Tickets:*"]
    for _, row in results.iterrows():
        issue_url = f"https://capillarytech.atlassian.net/browse/{row['Issue key']}"
        line = (
            f"• *<{issue_url}|{row['Issue key']}>* — Similarity: `{row['Similarity']:.2f}%`\n"
            f"   {row['Summary']}\n"
            f"   _{row['First 50% Comments (Most important context)'][:50]}..._"
        )
        lines.append(line)
    return "\n".join(lines)


# Main execution
if __name__ == "__main__":
    query = "Points are not allocated to the user in TATA"

    # Find matches
    matches = find_similar_tickets(query)
    print(matches)

    # Format Slack message
    message = format_results_for_slack(matches)
    slack_text = f"*Top matches for:* _{query}_\n\n{message}"

    # Send to Slack
    url = "https://v1.nocodeapi.com/harikishan/slack/IgAThybWcFBAkNHJ/sendText"
    params = {"text": slack_text}

    try:
        response = requests.post(url=url, params=params)
        response.raise_for_status()
        print("✅ Message sent to Slack.")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send message to Slack: {e}")
