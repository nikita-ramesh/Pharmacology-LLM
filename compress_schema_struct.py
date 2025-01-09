import json

def compress_json(input_file, output_file):
    """
    Compress a JSON file by shortening keys, removing unnecessary whitespace,
    and applying other optimizations.
    """
    # Load the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Compression strategies
    compressed_data = {"tables": []}
    table_lookup = {}  # Map table names to indices
    column_lookup = {}  # Map column names to indices
    table_id = 0
    column_id = 0

    for table in data.get("tables", []):
        table_name = table["table_name"]

        # Add the table to the lookup if not already present
        if table_name not in table_lookup:
            table_lookup[table_name] = table_id
            table_id += 1

        # Map columns to indices
        compressed_columns = []
        for column in table["columns"]:
            if column not in column_lookup:
                column_lookup[column] = column_id
                column_id += 1
            compressed_columns.append(column_lookup[column])

        # Add the compressed table
        compressed_data["tables"].append({
            "table_id": table_lookup[table_name],
            "columns": compressed_columns
        })

    # Add lookups for table names and column names
    compressed_data["table_lookup"] = {v: k for k, v in table_lookup.items()}
    compressed_data["column_lookup"] = {v: k for k, v in column_lookup.items()}

    # Save the compressed JSON
    with open(output_file, 'w') as f:
        json.dump(compressed_data, f, separators=(',', ':'))  # Minimize whitespace

    print(f"Compressed JSON saved to {output_file}")


if __name__ == "__main__":
    # Input and output file paths
    input_json_file = "schema_structure.json"
    output_json_file = "schema_structure_compressed.json"

    # Compress the JSON
    compress_json(input_json_file, output_json_file)
