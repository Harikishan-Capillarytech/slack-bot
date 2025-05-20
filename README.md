Pixie - Your Slack Assistant for Jira

Pixie✨ is a smart Slack-based assistant that helps you **find similar Jira tickets**, **predict their priority**, and even **create new issues** directly from Slack. Pixie also adds helpful context to tickets she creates — all powered by AI.

 ##Features

- `/pixie-similar`: Finds similar Jira tickets based on your query using sentence embeddings.
-  Predicts ticket **priority** using Gemini AI and a rules-based dataset.
- `/pixie-create`: Creates new Jira issues from Slack with enriched context and suggestions.
-  Automatically comments on new issues with related Jira tickets for quick triaging.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/pixie-jira-slackbot.git
cd pixie-jira-slackbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. .env file

GEMINI_API_TOKEN = token
SLACK_BOT_TOKEN = token
SLACK_APP_TOKEN = token
SLACK_SIGNING_SECRET = secret
SLACK_VERIFICATION_TOKEN = token

#jira tokens
JIRA_BASE_URL = https://capillarytech.atlassian.net
JIRA_EMAIL = email
JIRA_API_TOKEN= token
JIRA_PROJECT_KEY = CJ


### 4. Required files

jira_df_v2.pkl: Pre-processed Jira data.

jira_embeddings_v2.pkl: Sentence transformer embeddings of Jira tickets.

priority_rules.csv: Rules for AI-based priority prediction.

jira_cleaned_v2.csv: Cleaned ticket data


### 5. Generate pkl files

In embeddings.py, pass your cleaned csv file and modify the sentence transformer based on your requirement. 

### 6. Running locally

```bash
python3 mainJira.py
```

Also expose your server using ngrok
```bash
ngrok http 5000
```

### 7. Note

- Pixie uses Gemini API, so ensure your usage is within limits.

- jira_df_v2.pkl and jira_embeddings_v2.pkl should be updated periodically with recent Jira data.

- Pixie tags all issues she creates and adds context in comments for traceability and training feedback.


Author : Harikishan M - Capillary technologies

