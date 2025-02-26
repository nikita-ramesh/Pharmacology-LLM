import base64
import codecs
import requests
import psycopg2
import pandas as pd
import json
import openai
import getpass

# Function to generate the password dynamically
def pwd():
    return getpass.getpass(prompt="Enter the database password: ")

# Database connection details
db_config = {
    'host': 'localhost',
    'database': 'guide_to_pharmacology',
    'user': 'postgres',
    'password': pwd(),
}

# Function to connect to the PostgreSQL database
def connect_to_db():
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Function to train OpenAI with custom data
def train_openai_model(training_file):
    try:
        # Convert the DataFrame to JSONL format for OpenAI
        def convert_to_openai_format(df):
            training_data = []
            for _, row in df.iterrows():
                training_data.append({
                    "prompt": row['Natural Language Query'].strip() + "\n\n###\n\n",
                    "completion": " " + row['SQL'].strip() + " END"
                })
            return training_data

        # Load the training data
        train_df = pd.read_csv(training_file)
        training_data = convert_to_openai_format(train_df)

        # Save training data to a JSONL file
        jsonl_file = "training_data.jsonl"
        with open(jsonl_file, 'w') as f:
            for item in training_data:
                f.write(f"{json.dumps(item)}\n")

        # Upload training file to OpenAI
        response = openai.File.create(file=open(jsonl_file, 'rb'), purpose='fine-tune')
        file_id = response['id']
        print(f"Uploaded file. File ID: {file_id}")

        # Fine-tune the model
        fine_tune_response = openai.FineTune.create(training_file=file_id, model="davinci")
        print(f"Fine-tuning initiated: {fine_tune_response}")
    except Exception as e:
        print(f"Error during training: {e}")

# Function to ask a question and get SQL using OpenAI API
def ask_openai(question, additional_examples=None):
    api_key = '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'
    openai.api_key = api_key

    messages = [{"role": "system", "content": "You are a IUPHAR Guide to Pharmacology expert that generates SQL queries to the database for natural language queries"}]

    if additional_examples:
        messages.extend(additional_examples)

    # Add the user question as the prompt
    messages.append({"role": "user", "content": question})

    try:
        # Correct the model name to 'gpt-4' or 'gpt-3.5-turbo'
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=messages
        )

        message_content = response['choices'][0]['message']['content']

        if '```sql' in message_content:
            sql_query = message_content.split('```sql')[1].split('```')[0].strip()
            return sql_query
        else:
            print("No SQL query found in the response. Full response:")
            print(message_content)
            return None
    except Exception as e:
        print(f"Error extracting SQL query: {e}")
        return None

# Function to execute SQL query
def execute_query(conn, query):
    try:
        return pd.read_sql_query(query, conn)  # Use pandas to execute the query and return a DataFrame
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

# Main function
def main():
    # Connect to the database
    conn = connect_to_db()
    if not conn:
        return
    
    training_file = 'Training/all_queries_categorised_train.csv'  # Replace with the actual path
    train_openai_model(training_file)
    print("trained model")

    while True:
        # Ask the user for input
        question = input("Enter your question (or type 'exit' to quit): ")

        if question.lower() == 'exit':
            print("Exiting the program.")
            break

        sql_query = ask_openai(question)

        if sql_query:
            print(f"Generated SQL Query: {sql_query}")

            results = execute_query(conn, sql_query)
            if results is not None and not results.empty:
                print("Query Results:")
                print(results)
            else:
                print("No results found.")

    conn.close()

if __name__ == "__main__":
    main()