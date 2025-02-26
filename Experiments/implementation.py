import base64
import codecs
import requests
import psycopg2
import pandas as pd
import json
import random
import getpass

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

# Function to read schema from the JSON file
def load_schema_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            schema = json.load(file)
        return schema
    except Exception as e:
        print(f"Error loading schema: {e}")
        return None

# Function to load and split training data from CSV
def load_and_split_training_data(file_path, split_ratio=0.5, random_seed=42):
    try:
        df = pd.read_csv(file_path)
        
        # Ensure the dataset has the required columns
        if 'SQL' not in df.columns or 'Training/test set' not in df.columns:
            print("Error: Required columns not found in the dataset.")
            return None, None

        # Drop rows with missing values in relevant columns
        df = df[['SQL', 'Training/test set']].dropna()

        # Shuffle the dataset for randomness
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Calculate the split index
        split_index = int(len(df) * split_ratio)

        # Split into training and test sets
        training_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]

        print(f"Dataset split: {len(training_data)} training samples, {len(test_data)} test samples.")
        return training_data, test_data

    except Exception as e:
        print(f"Error loading training data: {e}")
        return None, None

# Function to generate schema string from the schema structure
def generate_schema_string(schema):
    schema_str = "The database consists of the following tables:\n"
    for table in schema["tables"]:
        table_name = table["table_name"]
        columns = ', '.join(table["columns"])
        schema_str += f"Table: {table_name}\nColumns: {columns}\n"
    return schema_str

# Function to initialize the schema context for the conversation
def initialize_schema_context(schema_str, training_data_sample):
    # Send the schema context along with some sample queries from the training data
    initial_message = {
        "role": "system", 
        "content": f"You have the following database schema:\n{schema_str}\nHere are some examples of natural language queries and their corresponding SQL queries:\n{training_data_sample}\n"
    }
    return initial_message

# Function to process user queries and generate SQL using OpenAI API
def process_user_query(question, schema_context):
    api_key = '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'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Add user query to the schema context
    data = {
        "model": "gpt-4o",
        "messages": schema_context + [{"role": "user", "content": question}]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code == 200:
        try:
            message_content = response.json()['choices'][0]['message']['content']
            
            # Try to extract the SQL query using the specific format
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
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to execute SQL query
def execute_query(conn, query):
    try:
        return pd.read_sql_query(query, conn)  # Use pandas to execute the query and return a DataFrame
    except Exception as e:
        print(f"Error executing query: {e}")
        return None  # Return None on error

def main():
    # Connect to the database
    conn = connect_to_db()
    if not conn:
        return

    # Load the schema structure from the JSON file
    schema = load_schema_from_file('schema_structure.json')
    if not schema:
        print("Failed to load schema.")
        return

    # Load and split training data
    training_data, test_data = load_and_split_training_data('Training/all_queries_categorised_train.csv', split_ratio=0.5)

    if training_data is None or training_data.empty:
        print("No training data available.")
        return

    # Sample all training queries and SQL pairs for context (use only from training data)
    training_data_sample = "\n".join([f"Q: {row['SQL']}\nA: {row['SQL']}" for _, row in training_data.iterrows()])

    # Generate a string representation of the schema
    schema_str = generate_schema_string(schema)
    print("Schema loaded successfully.")

    # Initialize the schema context for the conversation with sample training data
    schema_context = [initialize_schema_context(schema_str, training_data_sample)]  # Initial context

    while True:
        # Ask the user for a question
        question = input("Enter your question (or type 'exit' to quit): ")
        
        # Exit the loop if the user types 'exit'
        if question.lower() == 'exit':
            print("Exiting the program.")
            break

        # Process the user's query with the schema context
        sql_query = process_user_query(question, schema_context)

        if sql_query:
            print(f"Generated SQL Query: {sql_query}")

            # Execute the SQL query
            results = execute_query(conn, sql_query)
            if results is not None and not results.empty:
                print("Query Results:")
                print(results)  # Pandas will display the DataFrame in a structured format
            else:
                print("No results found.")

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()
