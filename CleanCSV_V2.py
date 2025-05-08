import pandas as pd
import math
import re
# === Step 1: Load the CSV ===
input_file = "Jira Export Excel CSV (all fields) 20250430230821.csv"  # Replace with your file path
df = pd.read_csv(input_file)



def clean_comment(comment):
    comment = str(comment).strip()

    # Remove all occurrences of Jira metadata panel lines
    comment = re.sub(
        r"^\d{1,2}/[A-Za-z]{3}/\d{2,4} \d{1,2}:\d{2} [AP]M;712020:[a-z0-9\-]+;\{panel:bgColor=#deebff\}*",
        "",
        comment,
        flags=re.IGNORECASE
    )
    comment = comment.replace("{panel}", "")
    # Remove mentions like @name or cc @name
    comment = re.sub(r"\bcc\s*@\w+\s*", "", comment, flags=re.IGNORECASE)
    comment = re.sub(r"@\w+", "", comment)

    # Remove "*Original Author: ...*" and "Posted on : ..."
    comment = re.sub(r"\*Original Author:.*?\*", "", comment, flags=re.IGNORECASE)
    comment = re.sub(r"Posted on\s*:\s*.*", "", comment, flags=re.IGNORECASE)

    # Remove common phrases
    remove_phrases = [
        r"^hi team[,!\.\s]*", r"^hello[,!\.\s]*", r"^hi[,!\.\s]*",
        r"^thanks[,!\.\s]*", r"^thank you[,!\.\s]*", r"^regards[,!\.\s]*",
        r"\bthanks\b", r"\bthank you\b", r"\bregards\b", r"\bcc\b"
    ]
    for pattern in remove_phrases:
        comment = re.sub(pattern, '', comment, flags=re.IGNORECASE)

    # Collapse multiple spaces and newlines
    comment = re.sub(r'\n+', '\n', comment)
    comment = re.sub(r'\s+', ' ', comment)

    return comment.strip()

# === Step 2: Select expected columns only ===
# Flexible match for key fields
expected_columns = ['Issue key', 'Issue id', 'Issue Type', 'Summary', 'Description']
component_cols = [col for col in df.columns if 'Components' in col]
comment_cols = [col for col in df.columns if 'Comment' in col]

# Verify required base columns exist
missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required column(s): {missing_cols}")

# === Step 3: Process each row ===
def process_row(row):
    # Merge Components
    components = [str(row[col]) for col in component_cols if pd.notna(row[col])]
    merged_components = ', '.join(components)

    # Get all non-empty comments
    comments = [
    clean_comment(str(row[col]))
    for col in comment_cols
    if pd.notna(row[col]) and str(row[col]).strip()
]

    # First 50% of comments
    n_first = max(1, math.ceil(len(comments) * 0.5))
    first_half = comments[:n_first]

    # Top 3 largest comments by length
    largest_comments = sorted(comments, key=len, reverse=True)[:3]

    # Remaining comments (excluding both above)
    remaining = [c for c in comments if c not in largest_comments and c not in first_half]


    return pd.Series({
        'Issue key': row['Issue key'],
        'Issue id': row['Issue id'],
        'Issue Type': row['Issue Type'],
        'Components': merged_components,
        'Summary': row['Summary'],
        'Description': row['Description'],
        'Largest comments (Most probable resolutions and most context)': '\n'.join(largest_comments),
        'First 50% Comments (Most important context)': '\n'.join(first_half),
        'Remaining Comments': '\n'.join(remaining)
    })

# === Step 4: Apply row processing ===
result_df = df.apply(process_row, axis=1)

# === Step 5: Save result ===
output_file = "jira_cleaned_v2.csv"
result_df.to_csv(output_file, index=False)

print(f"✅ Done! Processed file saved to: {output_file}")
