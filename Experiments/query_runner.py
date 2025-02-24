import json
import requests
import psycopg2
import pandas as pd
import getpass

def pwd():
    return getpass.getpass(prompt="Enter the database password: ")

db_config = {
    'host': 'localhost',
    'database': 'guide_to_pharmacology',
    'user': 'postgres',
    'password': pwd(),
}

def connect_to_db():
    try:
        conn = psycopg2.connect(**db_config)
        conn.set_client_encoding('UTF8')
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def load_test_data(test_file_path):
    try:
        df_test = pd.read_csv(test_file_path)
        if 'Natural Language Query' not in df_test.columns:
            print("Error: Required columns not found in the test dataset.")
            return None

        df_test = df_test[['ID', 'Natural Language Query']]
        return df_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def load_schema_context(input_path="schema_context.json"):
    try:
        with open(input_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading schema context: {e}")
        return None

def process_user_query(question, schema_context):
    api_key = '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'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": schema_context + [{"role": "user", "content": question}]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        try:
            data = response.json()
            message_content = response.json()['choices'][0]['message']['content']
            generated_sql = message_content.split('```sql')[1].split('```')[0].strip() if '```sql' in message_content else None
            return generated_sql
        except Exception as e:
            print(f"Error extracting SQL query: {e}")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def execute_query(conn, query):
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

def run_test_set():
    conn = connect_to_db()
    if not conn:
        return

    test_data = load_test_data('Training/all_queries_categorised_test.csv')
    if test_data is None or test_data.empty:
        print("Error: Testing data unavailable.")
        return

    schema_context = load_schema_context()
    if not schema_context:
        print("Error: Failed to load schema context.")
        return

    for index, row in test_data.iterrows():
        nlq = row['Natural Language Query']
        generated_sql = process_user_query(nlq, schema_context)
        if generated_sql:
            generated_results = execute_query(conn, generated_sql)
            print(f"Query {index+1}: {nlq}\nGenerated SQL:\n{generated_sql}\nResults:\n{generated_results}\n")

    conn.close()

if __name__ == "__main__":
    run_test_set()
