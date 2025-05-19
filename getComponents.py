import difflib
import os
import requests

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

def get_jira_components():
    url = f"{JIRA_BASE_URL}/rest/api/3/project/{JIRA_PROJECT_KEY}/components"
    response = requests.get(url, auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    if response.status_code == 200:
        return [comp["name"] for comp in response.json()]
    return []

def match_multiple_items(user_inputs, options):
    matched_items = []
    for user_input in user_inputs:
        match = difflib.get_close_matches(user_input.strip(), options, n=1, cutoff=0.5)
        if match:
            matched_items.append(match[0])
    return matched_items