import base64
import codecs
import requests
import psycopg2
import pandas as pd
import json
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
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def load_schema_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading schema: {e}")
        return None

def load_and_split_training_data(file_path, split_ratio=0.5, random_seed=42):
    try:
        df = pd.read_csv(file_path)
        if 'Natural Language Query' not in df.columns or 'SQL' not in df.columns or 'Training/test set' not in df.columns:
            print("Error: Required columns not found in the dataset.")
            return None, None

        df = df[['ID', 'Natural Language Query', 'SQL', 'Training/test set']].dropna()
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        split_index = int(len(df) * split_ratio)
        return df.iloc[:split_index], df.iloc[split_index:]
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None, None

def generate_schema_string(schema):
    schema_str = "The database consists of the following tables:\n"
    for table in schema["tables"]:
        schema_str += f"Table: {table['table_name']}\nColumns: {', '.join(table['columns'])}\n"
    return schema_str

def initialize_schema_context(schema_str, training_data_sample):
    return {
        "role": "system",
        "content": (
            f"You have the following PostgreSQL database schema:\n{schema_str}\n"
            f"Here are some examples of natural language queries and their corresponding PostgreSQL queries:\n{training_data_sample}\n"
            "Your task is to generate valid, executable PostgreSQL queries based on natural language questions.\n"
            "If one asks for a writeup generate SQL queries that retrieve relevant comments or info on those topics"
            "Generate only **fully functional SQL queries** without placeholders or missing clauses."
        )
    }

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
            message_content = response.json()['choices'][0]['message']['content']
            return message_content.split('```sql')[1].split('```')[0].strip() if '```sql' in message_content else None
        except Exception as e:
            print(f"Error extracting SQL query: {e}")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def execute_query(conn, query):
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

def run_test_set():
    conn = connect_to_db()
    if not conn:
        return

    schema = load_schema_from_file('schema_structure.json')
    if not schema:
        print("Failed to load schema.")
        return

    training_data, test_data = load_and_split_training_data('Training/all_queries_categorised_train.csv', split_ratio=0.5)
    if training_data is None or training_data.empty or test_data is None or test_data.empty:
        print("Error: Training or test data unavailable.")
        return

    training_data_sample = "\n".join([f"Q: {row['Natural Language Query']}\nA: {row['SQL']}" for _, row in training_data.iterrows()])
    schema_str = generate_schema_string(schema)
    schema_context = [initialize_schema_context(schema_str, training_data_sample)]

    success_count = 0
    total_count = 0
    none_set = set()
    empty_set = set()

    for index, row in test_data.iterrows():
        total_count += 1
        nlq = row['Natural Language Query']
        expected_sql = row['SQL']

        print(f"\n--- Processing Test Case {index + 1} ---")
        print(f"Natural Language Query: {nlq}")
        print(f"Expected SQL: {expected_sql}")

        generated_sql = process_user_query(nlq, schema_context)
        print(f"Generated SQL: {generated_sql}")

        expected_results = execute_query(conn, expected_sql) if expected_sql else None
        generated_results = execute_query(conn, generated_sql) if generated_sql else None

        if generated_results is not None and not generated_results.empty:
            success_count += 1
        
        if generated_results is None:
            none_set.add(int(row['ID']))
        elif generated_results.empty:
            empty_set.add(int(row['ID']))

        print("\nExpected SQL Results:")
        print(expected_results if expected_results is not None else "Error executing expected SQL.")

        print("\nGenerated SQL Results:")
        print(generated_results if generated_results is not None else "Error executing generated SQL.")

    conn.close()

    print("Total test queries executed:", total_count)
    print("Successful test queries:", success_count)
    print("None count:", len(none_set), none_set)
    print("Empty count:", len(empty_set), empty_set)

if __name__ == "__main__":
    run_test_set()
