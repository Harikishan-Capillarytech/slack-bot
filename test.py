import joblib
import numpy as np

priority_embeddings = joblib.load("priority_rules_openai_embeddings.pkl")
print("✅ Dim of first embedding:", len(priority_embeddings[0]))