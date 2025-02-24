import pandas as pd
import json

def load_training_data(file_path):
    """Loads the training data from a CSV file and checks for required columns."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Natural Language Query', 'Notes for student', 'SQL', '2nd SQL', 'Minimum output columns']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        
        return df[required_columns]  # Return only required columns
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

def prepare_fine_tuning_data(df, output_file):
    """Prepares and saves the fine-tuning data in JSONL format."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                json_entry = {
                    "prompt": row['Natural Language Query'],  # Only the NL query
                    "completion": (
                        f"Notes: {row['Notes for student']}\n"
                        f"Min required cols: {row['Minimum output columns']}\n"
                        f"Answer: {row['SQL']}\n"
                        f"Alternative Answer: {row['2nd SQL']}"
                        f" <|endoftext|>"
                    )
                }
                f.write(json.dumps(json_entry) + '\n')
        
        print(f"Fine-tuning data saved to {output_file}")
    except Exception as e:
        print(f"Error preparing fine-tuning data: {e}")

if __name__ == "__main__":
    input_csv = 'Training/all_queries_categorised_train.csv'  # Path to your CSV file
    output_jsonl = 'fine_tuning_data.jsonl'  # Output JSONL file path

    training_data = load_training_data(input_csv)
    if training_data is not None:
        prepare_fine_tuning_data(training_data, output_jsonl)
