import pandas as pd
import re

# Load your CSV file
df = pd.read_csv("Jira Export Excel CSV (all fields) 20250430230821.csv")

# --- Step 1: Keep only necessary columns ---
# Find all columns related to comments
comment_cols = [col for col in df.columns if 'comment' in col.lower()]
fields_to_keep = ['Issue key','Issue id', 'Issue Type','Summary', 'Description', 'Components'] + comment_cols
df = df[[col for col in fields_to_keep if col in df.columns]]

# --- Step 2: Combine text fields into a single text column ---
# Fill NaNs with empty strings first
for col in ['Description'] + comment_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

# Combine all text fields into one
df['Combined_Text'] = df[['Description'] + comment_cols].agg(' '.join, axis=1)

# --- Step 3: Clean text ---
def clean_text(text):
    # Remove mentions like @username
    text = re.sub(r'@\w+', '', text)
    
    # Remove common email-style greetings or sign-offs
    text = re.sub(r'(?i)\bhi team\b', '', text)
    text = re.sub(r'(?i)\bthanks\b', '', text)
    text = re.sub(r'(?i)\bthank you\b', '', text)
    text = re.sub(r'(?i)\bregards\b.*', '', text)
    text = re.sub(r'(?i)\bcheers\b.*', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text




df['Cleaned_Text'] = df['Combined_Text'].apply(clean_text)

# --- Step 4: Save the cleaned data ---
df_cleaned = df[['Issue key','Issue id', 'Issue Type', 'Summary', 'Components', 'Cleaned_Text']]
df_cleaned.to_csv("jira_cleaned.csv", index=False)


print("✅ Cleaned CSV saved as 'jira_cleaned.csv'")
