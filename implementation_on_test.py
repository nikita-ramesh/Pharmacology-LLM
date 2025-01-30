import base64
import codecs
import requests
import psycopg2
import pandas as pd
import json

def pwd():
    s1 = ''.join([chr(int(i)) for i in ['120', '65', '103', '108', '101', '116', '116', '55']])
    s2 = base64.b64encode(s1.encode('utf-8')).decode('utf-8')
    s3 = codecs.encode(s2[::-1], 'rot_13')
    s4 = codecs.decode(s3[::-1], 'rot_13')
    return base64.b64decode(s4).decode('utf-8')

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

        df = df[['Natural Language Query', 'SQL', 'Training/test set']].dropna()
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
        "content": f"You have the following database schema:\n{schema_str}\n"
                   f"Here are some examples of natural language queries and their corresponding SQL queries:\n{training_data_sample}\n"
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

    for index, row in test_data.iterrows():
        nlq = row['Natural Language Query']
        expected_sql = row['SQL']

        print(f"\n--- Processing Test Case {index + 1} ---")
        print(f"Natural Language Query: {nlq}")
        print(f"Expected SQL: {expected_sql}")

        generated_sql = process_user_query(nlq, schema_context)
        print(f"Generated SQL: {generated_sql}")

        expected_results = execute_query(conn, expected_sql) if expected_sql else None
        generated_results = execute_query(conn, generated_sql) if generated_sql else None

        print("\nExpected SQL Results:")
        print(expected_results if expected_results is not None else "Error executing expected SQL.")

        print("\nGenerated SQL Results:")
        print(generated_results if generated_results is not None else "Error executing generated SQL.")

    conn.close()

if __name__ == "__main__":
    run_test_set()
