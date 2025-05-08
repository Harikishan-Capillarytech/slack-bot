import pandas as pd

# Load CSV
df = pd.read_csv("Jira Export Excel CSV (all fields) 20250430230821.csv")

# Show shape and preview
# print("Shape:", df.shape)
# print(df.columns)
df.head()

# Drop rows with missing summaries or issue types
df = df.dropna(subset=['Summary', 'Issue Type'])

# Optional: Fill missing descriptions with empty string
df['Description'] = df['Description'].fillna('')

# Combine Summary and Description
df['Text'] = df['Summary'] + ' ' + df['Description']

# Check class distribution
print(df['Issue Type'].value_counts())