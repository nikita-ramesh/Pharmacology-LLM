import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from demo_1 import connect_to_db

# Load the dataset
file_path = "Training/all_queries_categorised_train.csv"  # Path to your training set file
df = pd.read_csv(file_path)

# Splitting the dataset
def split_dataset(df):
    # Identify difficulty levels
    difficulty_cols = ["Difficulty: Easy", "Difficulty: Easy-Moderate", "Difficulty: Moderate-Hard", "Difficulty: Hard"]
    df["Difficulty"] = df[difficulty_cols].idxmax(axis=1).str.replace("Difficulty: ", "")
    
    # Stratified split by difficulty
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Difficulty"], random_state=42)
    return train_df, test_df

train_df, test_df = split_dataset(df)

# Function to execute a query and return results
def execute_query(conn, query):
    try:
        return pd.read_sql_query(query, conn)  # Use pandas to execute the query and return a DataFrame
    except Exception as e:
        print(f"Error executing query: {e}")
        return None  # Return None on error

# Evaluation metrics
def evaluate_query(expected_df, result_df, min_columns):
    if expected_df is None or result_df is None:
        return {"Precision": 0, "Recall": 0, "Accuracy": 0, "Missed Columns": min_columns, "Extra Columns": 0}
    
    expected_columns = set(expected_df.columns)
    result_columns = set(result_df.columns)
    
    # Compute missed and extra columns
    missed_columns = expected_columns - result_columns
    extra_columns = result_columns - expected_columns
    
    # Precision, Recall, Accuracy
    true_positive = len(expected_columns & result_columns)
    precision = true_positive / len(result_columns) if result_columns else 0
    recall = true_positive / len(expected_columns) if expected_columns else 0
    accuracy = int(len(missed_columns) == 0 and len(extra_columns) == 0)
    
    return {
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "Missed Columns": len(missed_columns),
        "Extra Columns": len(extra_columns),
    }

# Main evaluation loop
def evaluate_dataset(df, conn):
    metrics = []
    for _, row in df.iterrows():
        sql_query = row["SQL"]
        second_sql = row.get("2nd SQL")
        min_columns = row.get("Minimum output columns", 0)
        
        # Execute the main SQL query
        expected_df = execute_query(conn, sql_query)
        
        # Execute the second SQL query (if provided)
        second_expected_df = execute_query(conn, second_sql) if pd.notnull(second_sql) else None
        
        # Evaluate the results
        result_df = execute_query(conn, sql_query)
        main_eval = evaluate_query(expected_df, result_df, min_columns)
        
        # If second SQL exists, evaluate against it
        if second_expected_df is not None:
            second_eval = evaluate_query(second_expected_df, result_df, min_columns)
            # Take the better evaluation as the final metric
            for key in main_eval:
                main_eval[key] = max(main_eval[key], second_eval[key])
        
        # Append metrics for the query
        metrics.append({
            "Query ID": row["ID"],
            "Precision": main_eval["Precision"],
            "Recall": main_eval["Recall"],
            "Accuracy": main_eval["Accuracy"],
            "Missed Columns": main_eval["Missed Columns"],
            "Extra Columns": main_eval["Extra Columns"],
        })
    
    return pd.DataFrame(metrics)

# Connect to the database
conn = connect_to_db()
if conn:
    # Evaluate the training set
    train_metrics = evaluate_dataset(train_df, conn)
    print("Training Metrics:")
    print(train_metrics)
    
    # Evaluate the test set
    test_metrics = evaluate_dataset(test_df, conn)
    print("Test Metrics:")
    print(test_metrics)
    
    # Close the database connection
    conn.close()
