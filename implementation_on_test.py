import base64
import codecs
import requests
import psycopg2
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score
import numpy as np

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
    
# EVALUATION FUNCTIONS

def evaluate_execution_accuracy(expected_results, generated_results):
    """Evaluate whether the predicted result matches the gold result."""
    if expected_results is None or generated_results is None or expected_results.empty or generated_results.empty:
        return False
    return expected_results.equals(generated_results)

def evaluate_precision(expected_results, generated_results):
    """Calculate the precision of the generated query's results."""
    if expected_results is None or generated_results is None or expected_results.empty or generated_results.empty:
        return 0
    
    # Ensure both DataFrames have the same number of rows by aligning them
    expected_results = expected_results.dropna()
    generated_results = generated_results.dropna()

    # Check if the lengths match after dropping missing values
    if len(expected_results) != len(generated_results):
        print(f"Warning: Mismatched row counts. Expected: {len(expected_results)}, Generated: {len(generated_results)}")
        return 0

    # Flatten both results to compare as arrays
    expected = expected_results.values.flatten()
    predicted = generated_results.values.flatten()

    # Ensure arrays have the same length after flattening
    if len(expected) != len(predicted):
        print(f"Warning: Mismatched flattened lengths. Expected: {len(expected)}, Predicted: {len(predicted)}")
        return 0

    # Calculate True Positives, False Positives, and False Negatives
    TP = sum((expected == 1) & (predicted == 1))  # Correctly predicted positive
    FP = sum((expected == 0) & (predicted == 1))  # Incorrectly predicted positive
    FN = sum((expected == 1) & (predicted == 0))  # Incorrectly predicted negative

    # Avoid division by zero
    if TP + FP == 0:
        return 0
    precision = TP / (TP + FP) * 100
    return precision


def evaluate_recall(expected_results, generated_results):
    """Calculate the recall of the generated query's results."""
    if expected_results is None or generated_results is None or expected_results.empty or generated_results.empty:
        return 0
    
    # Ensure both DataFrames have the same number of rows by aligning them
    expected_results = expected_results.dropna()
    generated_results = generated_results.dropna()

    # Check if the lengths match after dropping missing values
    if len(expected_results) != len(generated_results):
        print(f"Warning: Mismatched row counts. Expected: {len(expected_results)}, Generated: {len(generated_results)}")
        return 0

    # Flatten both results to compare as arrays
    expected = expected_results.values.flatten()
    predicted = generated_results.values.flatten()

    # Ensure arrays have the same length after flattening
    if len(expected) != len(predicted):
        print(f"Warning: Mismatched flattened lengths. Expected: {len(expected)}, Predicted: {len(predicted)}")
        return 0

    # Calculate True Positives and False Negatives
    TP = sum((expected == 1) & (predicted == 1))  # Correctly predicted positive
    FN = sum((expected == 1) & (predicted == 0))  # Incorrectly predicted negative

    # Avoid division by zero
    if TP + FN == 0:
        return 0
    recall = TP / (TP + FN) * 100
    return recall

def evaluate_column_accuracy(expected_results, generated_results):
    """Calculate the column accuracy of the generated query's results."""
    if expected_results is None or generated_results is None or expected_results.empty or generated_results.empty:
        return 0
    expected_columns = set(expected_results.columns)
    generated_columns = set(generated_results.columns)
    correct_columns = expected_columns.intersection(generated_columns)
    return (len(correct_columns) / len(expected_columns)) * 100 if len(expected_columns) > 0 else 0

def evaluate_extra_missing_columns(expected_results, generated_results):
    """Evaluate the number of extra and missing columns in the generated result."""
    if expected_results is None or generated_results is None or expected_results.empty or generated_results.empty:
        return 0, 0
    expected_columns = set(expected_results.columns)
    generated_columns = set(generated_results.columns)
    extra_columns = len(generated_columns - expected_columns)
    missing_columns = len(expected_columns - generated_columns)
    return extra_columns, missing_columns

def run_evaluation(expected_sql, generated_sql, expected_results, generated_results, query_index):
    """Evaluate a pair of queries and generate a result summary."""
    execution_accuracy = evaluate_execution_accuracy(expected_results, generated_results)
    # exact_accuracy = evaluate_exact_accuracy(expected_sql, generated_sql)
    precision = evaluate_precision(expected_results, generated_results)
    recall = evaluate_recall(expected_results, generated_results)
    column_accuracy = evaluate_column_accuracy(expected_results, generated_results)
    extra_columns, missing_columns = evaluate_extra_missing_columns(expected_results, generated_results)
    
    # Print results for the query pair
    print(f"\nResults for Query Pair {query_index + 1}:")
    print(f"  Execution Accuracy: {execution_accuracy}")
    # print(f"  Exact Accuracy: {exact_accuracy}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  Column Accuracy: {column_accuracy:.2f}%")
    print(f"  Extra Columns: {extra_columns}")
    print(f"  Missing Columns: {missing_columns}")
    
    return execution_accuracy, precision, recall, column_accuracy, extra_columns, missing_columns

def evaluate_overall(accuracies):
    """Calculate overall evaluation metrics."""
    total_pairs = len(accuracies)
    execution_accuracy = np.mean([acc[0] for acc in accuracies]) * 100
    exact_accuracy = np.mean([acc[1] for acc in accuracies]) * 100
    precision = np.mean([acc[2] for acc in accuracies])
    recall = np.mean([acc[3] for acc in accuracies])
    column_accuracy = np.mean([acc[4] for acc in accuracies])
    avg_extra_columns = np.mean([acc[5] for acc in accuracies])
    avg_missing_columns = np.mean([acc[6] for acc in accuracies])
    
    print("\nOverall Accuracy Percentages:")
    print(f"  Execution Accuracy: {execution_accuracy:.2f}%")
    print(f"  Exact Accuracy: {exact_accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    print(f"  Column Accuracy: {column_accuracy:.2f}%")
    print(f"  Average Extra Columns per Query Pair: {avg_extra_columns:.2f}")
    print(f"  Average Missing Columns per Query Pair: {avg_missing_columns:.2f}")
    
    return execution_accuracy, exact_accuracy, precision, recall, column_accuracy, avg_extra_columns, avg_missing_columns

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
    
    accuracies = []

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

        # Evaluate the results
        acc = run_evaluation(expected_sql, generated_sql, expected_results, generated_results, index)
        accuracies.append(acc)

    conn.close()

    # Calculate overall accuracy
    evaluate_overall(accuracies)

if __name__ == "__main__":
    run_test_set()
