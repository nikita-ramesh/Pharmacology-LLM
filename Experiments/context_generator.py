import json
import pandas as pd

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

        df = df[['ID', 'Natural Language Query', 'SQL', 'Greek', '2nd SQL', 'Training/test set']].dropna()
        return df
    except Exception as e:
        print(f"Error loading training data: {e}")
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

def save_schema_context(schema_context, output_path="../schema_context.json"):
    try:
        with open(output_path, "w") as file:
            json.dump(schema_context, file)
        print(f"Schema context saved to {output_path}")
    except Exception as e:
        print(f"Error saving schema context: {e}")

def generate_and_cache_schema():
    schema = load_schema_from_file('../schema_structure.json')
    if not schema:
        print("Failed to load schema.")
        return

    training_data = load_training_data('Training/all_queries_categorised_train.csv')

    if training_data is None or training_data.empty:
        print("Error: Training data unavailable.")
        return
    
    training_data_sample = "\n".join([
        f"Q: {row['Natural Language Query']}\nA: {row['SQL']}\nAlternative A: {row['2nd SQL']}" 
        for _, row in training_data.iterrows()
    ])

    schema_str = generate_schema_string(schema)
    schema_context = [initialize_schema_context(schema_str, training_data_sample)]

    save_schema_context(schema_context)

if __name__ == "__main__":
    generate_and_cache_schema()
