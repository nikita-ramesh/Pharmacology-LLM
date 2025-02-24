import requests
import datetime

def check_usage(api_key, total_budget=2_000_000):
    base_url = "https://api.openai.com/v1/usage"

    # Get today's date in the required format
    today = datetime.date.today().strftime("%Y-%m-%d")

    params = {
        "date": today  # Single date, not a range
    }

    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()

        # Extract total tokens used
        context_tokens = sum(item.get("n_context_tokens_total", 0) for item in data.get("data", []))
        generated_tokens = sum(item.get("n_generated_tokens_total", 0) for item in data.get("data", []))
        
        total_used = context_tokens + generated_tokens
        remaining_tokens = total_budget - total_used

        print(f"Total Context Tokens: {context_tokens}")
        print(f"Total Generated Tokens: {generated_tokens}")
        print(f"Total Tokens Used: {total_used}")
        print(f"Remaining Tokens: {remaining_tokens}")
        
    else:
        print(f"Error fetching usage: {response.status_code} - {response.text}")

# Replace with your actual API key
api_key = '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'
check_usage(api_key)
