import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# Load CSV
priority_csv_path = "priority_rules.csv"  # change if needed
priority_df = pd.read_csv(priority_csv_path)

# Replace NaN with empty strings
priority_df.fillna("", inplace=True)

# Combine key rule fields into a string for embedding
def row_to_text(row):
    return f"{row['Feature/Product']} {row['UseCase']} {row['Component']}"

priority_df['rule_text'] = priority_df.apply(row_to_text, axis=1)

# Load model (you can reuse the same one from your app)
model = SentenceTransformer("all-MPNET-base-v2")

# Generate embeddings
print("🔍 Generating embeddings for priority rules...")
priority_embeddings = model.encode(priority_df['rule_text'].tolist(), normalize_embeddings=True)
print("✅ Embeddings complete.")

# Save for future use
joblib.dump(priority_df, "priority_rules_df.pkl")
joblib.dump(priority_embeddings, "priority_rules_embeddings.pkl")
print("💾 Saved rules and embeddings.")
