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

def load_training_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Natural Language Query' not in df.columns or 'SQL' not in df.columns or 'Training/test set' not in df.columns:
            print("Error: Required columns not found in the dataset.")
            return None

        df = df[['ID', 'Natural Language Query', 'SQL', 'Training/test set', 'Notes for student', '2nd SQL', 'Minimum output columns']]
        return df
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

def load_test_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Natural Language Query' not in df.columns or 'SQL' not in df.columns or 'Training/test set' not in df.columns:
            print("Error: Required columns not found in the dataset.")
            return None

        df = df[['ID', 'Natural Language Query', 'SQL', 'Training/test set']].dropna()
        return df
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

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
            "Follow these strict rules:\n"
            "1. Always define `WITH RECURSIVE` at the **beginning** of the query if needed.\n"
            "2. Ensure all subqueries are correctly nested and avoid placing `WITH RECURSIVE` inside `IN (...)` clauses.\n"
            "3. Always use `EXPLAIN` to verify the structure before finalizing the query.\n"
            "4. Avoid unnecessary parentheses that can cause syntax errors.\n"
            "5. Make sure column names match the provided schema exactly.\n"
            "6. When using `ILIKE`, ensure it applies to a valid text column and make the match **more flexible**:\n"
            "   - If filtering for a **single word** (e.g., 'delta'), use `ILIKE '%delta%' OR ILIKE '%Delta%'`.\n"
            "   - If filtering for **multi-word terms** (e.g., 'GABA B1' or 'GABA<sub>B1</sub>'), try **multiple formats**:\n"
            "     - `ILIKE '%GABAB1%' OR ILIKE '%GABA B1%' OR ILIKE '%GABA<sub>B1</sub>%'`.\n"
            "7. Use table aliases to make queries cleaner and ensure correct joins.\n"
            "8. When asked to do a writeup or essay, only generate PostgreSQL queries to gather as much relevant data as possible.\n"
            "Your task is to generate valid, executable PostgreSQL queries based on natural language questions.\n"
            "Generate only **fully functional SQL queries** without placeholders or missing clauses.\n"
        )
    }

def process_user_query(question, schema_context, error_message=None):
    api_key = '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'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Include previous error message if it exists
    if error_message:
        question = f"{error_message}\n{question}"

    data = {
        "model": "gpt-4o",
        "messages": schema_context + [{"role": "user", "content": question}]
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code == 200:
        try:
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
        results = pd.read_sql_query(query, conn)
        if results.empty:
            return results, "SQL executed but returned an empty result. Please check your query strings."
        return results, None
    except Exception as e:
        print(f"Error executing query: {e}")
        return None, str(e)

def execution_accuracy(predicted_df, gold_df):
    """
    Checks if the rows of two DataFrames are equal by comparing their sets of rows.
    """
    if predicted_df is not None and gold_df is not None:
        try:
            # Convert DataFrames to sets of tuples representing rows
            predicted_set = set(map(tuple, predicted_df.to_numpy()))
            gold_set = set(map(tuple, gold_df.to_numpy()))
            
            # Check if the sets of rows are equal
            return predicted_set == gold_set
        except Exception as e:
            print(f"Error comparing DataFrames: {e}")
            return False
    return False

def partial_execution_accuracy(predicted_df, gold_df):
    """
    Checks if there is at least one common column between predicted_df and gold_df.
    - If common columns exist, checks if all the rows are common in the filtered dataframes.
    - If no common columns exist, checks if all rows in predicted_df matches a row in gold_df completely.
    """
    if predicted_df is not None and gold_df is not None:
        try:
            # Find common columns
            common_columns = set(predicted_df.columns) & set(gold_df.columns)

            if common_columns:
                # Filter both DataFrames to only include the common columns
                filtered_predicted_df = predicted_df[list(common_columns)]
                filtered_gold_df = gold_df[list(common_columns)]

                # Convert rows in the filtered DataFrames to sets for comparison
                predicted_rows = {tuple(row) for row in filtered_predicted_df.itertuples(index=False, name=None)}
                gold_rows = {tuple(row) for row in filtered_gold_df.itertuples(index=False, name=None)}

            else:
                # No common columns, compare full rows without considering columns
                predicted_rows = {tuple(row) for row in predicted_df.itertuples(index=False, name=None)}
                gold_rows = {tuple(row) for row in gold_df.itertuples(index=False, name=None)}

            # Check if there is at least one row in common
            return predicted_rows == gold_rows # True if there's an intersection

        except Exception as e:
            print(f"Error during partial execution accuracy check: {e}")
            return False

    return False
    
def partial_col_accuracy(predicted_df, gold_df):
    """
    Checks if there is at least one common column between the predicted DataFrame and the gold DataFrame.
    """
    if predicted_df is not None and gold_df is not None:
        try:
            # Get the sets of column names.
            predicted_columns = set(predicted_df.columns)
            gold_columns = set(gold_df.columns)

            # Check if there is any intersection between the columns.
            return bool(predicted_columns & gold_columns)  # Returns True if intersection is not empty
        except Exception as e:
            print(f"Error during partial column accuracy check: {e}")
            return False
    return False

def run_test_set():
    conn = connect_to_db()
    if not conn:
        return

    schema = load_schema_from_file('schema_structure.json')
    if not schema:
        print("Failed to load schema.")
        return

    # Load 100% of training data
    training_data = load_training_data('Training/all_queries_categorised_train.csv')
    
    # Load test data from a separate file
    test_data = load_test_data('Training/all_queries_categorised_test.csv')

    if training_data is None or training_data.empty or test_data is None or test_data.empty:
        print("Error: Training or test data unavailable.")
        return

    training_data_sample = "\n".join([
        f"Q: {row['Natural Language Query']}\n"
        f"Notes: {row['Notes for student']}\n"
        f"A: {row['SQL']}\n"
        f"Alternative A: {row['2nd SQL']}"
        f"Min required cols: {row['Minimum output columns']}"
        for _, row in training_data.iterrows()
    ])
    schema_str = generate_schema_string(schema)
    schema_context = [initialize_schema_context(schema_str, training_data_sample)]

    success_count = 0
    total_count = 0
    exec_accurate_count = 0
    partial_exec_accurate_count = 0
    partial_cols_accurate_count = 0
    none_set = set()
    empty_set = set()

    # Open a file to write results
    with open("query_results.txt", "w", encoding="utf-8") as result_file:
        result_file.write("Test Set Evaluation Results\n")
        
        # Loop over test data to process each query
        for index, row in test_data.iterrows():
            total_count += 1
            nlq = row['Natural Language Query']
            expected_sql = row['SQL']

            result_file.write(f"\n--- Processing Test Case {index + 1} ---\n")
            result_file.write(f"Natural Language Query: {nlq}\n")
            result_file.write(f"Expected SQL: {expected_sql}\n")

            retries = 0
            max_retries = 1
            generated_results = None
            error_message = None

            while retries <= max_retries:
                # Process the user query with the current error message as context
                generated_sql = process_user_query(nlq, schema_context, error_message)

                if generated_sql is not None:
                    generated_results, error_message = execute_query(conn, generated_sql)

                    # If generated_results is None or empty, we'll retry
                    if generated_results is None or generated_results.empty:
                        retries += 1
                    else:
                        break  # Exit the loop if results are found

            # generated_results will now hold either the valid DataFrame, an empty DataFrame, or None

            result_file.write(f"Generated SQL: {generated_sql}\n")
            expected_results, _ = execute_query(conn, expected_sql) if expected_sql else (None, None)

            # Add to sets based on execution results
            if generated_results is None:
                none_set.add(int(row['ID']))
            elif generated_results.empty:
                empty_set.add(int(row['ID']))
            else:
                success_count += 1
            
            execution_accuracy_bool = execution_accuracy(generated_results, expected_results)
            partial_execution_accuracy_bool = partial_execution_accuracy(generated_results, expected_results)
            partial_col_accuracy_bool = partial_col_accuracy(generated_results, expected_results)

            if execution_accuracy_bool:
                exec_accurate_count += 1
            if partial_execution_accuracy_bool:
                partial_exec_accurate_count += 1
            if partial_col_accuracy_bool:
                partial_cols_accurate_count += 1

            result_file.write("\nExpected SQL Results:\n")
            result_file.write(f"{expected_results if expected_results is not None else 'Error executing expected SQL.'}\n")

            result_file.write("\nGenerated SQL Results:\n")
            result_file.write(f"{generated_results if generated_results is not None else 'Error executing generated SQL.'}\n")

            result_file.write(f"\nExecution Accuracy: {execution_accuracy_bool}\n")
            result_file.write(f"Partial Execution Accuracy: {partial_execution_accuracy_bool}\n")
            result_file.write(f"Partial Column Accuracy: {partial_col_accuracy_bool}\n")

        # Final statistics
        result_file.write("\n" + "="*50 + "\n")

        # Successful Statistics
        result_file.write("SUCCESSFUL STATISTICS:\n")
        result_file.write(f"Total test queries executed: {total_count}\n")
        result_file.write(f"Non empty output rate: {success_count/total_count}\n")
        result_file.write(f"Successful execution rate: {(success_count + len(empty_set)) / total_count}\n")
        result_file.write(f"Non empty test queries: {success_count}\n\n")

        # Detailed Execution Statistics
        result_file.write("DETAILED EXECUTION STATISTICS:\n")
        result_file.write(f"Execution Accuracy: {exec_accurate_count/total_count}\n")
        result_file.write(f"Partial Execution Accuracy: {partial_exec_accurate_count/total_count}\n")
        result_file.write(f"Partial Column Accuracy: {partial_cols_accurate_count/total_count}\n\n")

        # Failure Statistics
        result_file.write("FAILURE STATISTICS:\n")
        result_file.write(f"None count: {len(none_set)}\n")
        result_file.write(f"None IDs: {', '.join(map(str, none_set))}\n")
        result_file.write(f"Empty count: {len(empty_set)}\n")
        result_file.write(f"Empty IDs: {', '.join(map(str, empty_set))}\n")
        result_file.write("="*50 + "\n")

    conn.close()
    print("Results have been written to 'query_results.txt'.")
    
if __name__ == "__main__":
    run_test_set()
