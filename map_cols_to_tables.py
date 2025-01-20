import json

def map_columns_to_tables(input_file, output_file):
    """
    Map each column to all tables it exists in, and save the resulting structure to a new JSON file.
    """
    # Load the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create a mapping of columns to tables
    column_to_tables = {}

    for table in data.get("tables", []):
        table_name = table["table_name"]

        for column in table["columns"]:
            if column not in column_to_tables:
                column_to_tables[column] = []
            column_to_tables[column].append(table_name)

    # Create the output JSON structure
    output_data = {"columns": column_to_tables}

    # Save the output JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)  # Pretty-print the JSON with indentation

    print(f"Column-to-tables mapping saved to {output_file}")


if __name__ == "__main__":
    # Input and output file paths
    input_json_file = "schema_structure.json"
    output_json_file = "column_to_tables_mapping.json"

    # Generate the mapping
    map_columns_to_tables(input_json_file, output_json_file)
