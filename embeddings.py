from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib

# Load cleaned data
df = pd.read_csv("jira_cleaned_v2.csv")

df['combined_text'] = df['Summary'].fillna("") + " " + df['Description'].fillna("") + " " + df['Largest comments (Most probable resolutions and most context)'].fillna("")+ df['First 50% Comments (Most important context)'].fillna("") + " " + df['Remaining Comments'].fillna("")


# Use only the cleaned text for embeddings
texts = df['combined_text'].fillna("").tolist()

# Load a pre-trained model (small and efficient)
model = SentenceTransformer('all-MPNET-base-v2')  

# Generate embeddings for all tickets
embeddings = model.encode(texts, show_progress_bar=True)

joblib.dump(embeddings, "jira_embeddings_v2.pkl")
joblib.dump(df, "jira_df_v2.pkl")
print("✅ Embeddings and dataframe saved.")