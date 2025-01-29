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

# Function to generate a schema string from the schema structure
def generate_schema_string(schema):
    schema_str = "The database consists of the following tables:\n"
    for table in schema["tables"]:
        table_name = table["table_name"]
        columns = ', '.join(table["columns"])
        schema_str += f"Table: {table_name}\nColumns: {columns}\n"
    return schema_str

# Function to ask a question and get SQL using OpenAI API
def ask_openai(question, schema_str):
    api_key = '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'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": f"You have the following database schema:\n{schema_str}\n"},
            {"role": "user", "content": question}
        ]
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
                # If the SQL query format is missing, print the entire response
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

# Main function
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

    # Generate a string representation of the schema
    schema_str = generate_schema_string(schema)
    print("Schema loaded successfully.")

    while True:
        # Ask the user for a question
        question = input("Enter your question (or type 'exit' to quit): ")
        
        # Exit the loop if the user types 'exit'
        if question.lower() == 'exit':
            print("Exiting the program.")
            break

        # Generate SQL query based on the question and schema
        sql_query = ask_openai(question, schema_str)

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
