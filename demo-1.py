import base64
import codecs
import requests
import psycopg2
import pandas as pd

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

# Function to ask a question and get SQL using OpenAI API
def ask_openai(question):
    api_key = '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'  # Replace with your actual API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": '''
            I have a database with the following schema:
            
            Table: ‘object’ with columns: (abbreviation, only_grac, in_cgtp, annotation_status,
            comments, object_id, only_iuphar, cite_id, systematic_name, gtip_comment, name,
            gtmp_comment, in_gtip, old_object_id, in_gtmp, grac_comments, last_modified,
            no_contributor_list, structural_info_comments, quaternary_structure_comments).
            
            Table: ‘ligand’ with columns: (old_ligand_id, has_chembl_interaction,
            name_vector, mechanism_of_action_vector, comments, immuno_comments, antibacterial,
            ligand_id, bioactivity_comments, absorption_distribution_vector, pubchem_sid,
            metabolism_vector, comments_vector, popn_pharmacokinetics, bioactivity_comments_vector,
            elimination, drugs_url, abbreviation, absorption_distribution, organ_function_impairment,
            emc_url, verified, mechanism_of_action, approved_source, who_essential, abbreviation_vector,
            in_gtmp, metabolism, type, elimination_vector, approved, in_gtip, labelled, withdrawn_drug,
            name, immuno_comments_vector, popn_pharmacokinetics_vector, ema_url, iupac_name,
            clinical_use_vector, gtmp_comments, radioactive, has_qi_interaction,
            gtmp_comments_vector, organ_function_impairment_vector, clinical_use).
            
            Table: ‘interaction’ with columns: (affinity_high, affinity_low, concentration_range,
            original_affinity_relation, action_comment, assay_conditions, object_id,
            original_affinity_units, endogenous, type_vector, original_affinity_high_nm,
            action, affinity_median, voltage_dependent, assay_description, hide,
            from_grac, whole_organism_assay, only_grac, original_affinity_low_nm,
            percent_activity, affinity_units, affinity_voltage_median, primary_target, selectivity,
            use_dependent, species_id, selective, ligand_id, target_ligand_id, ligand_context,
            affinity_voltage_high, receptor_site, affinity_voltage_low, assay_url,
            original_affinity_median_nm, interaction_id, rank, affinity_physiological_voltage, type).
            
            Please write an SQL query based on the following natural language text:
            If anything is asked outside of this, don't answer.
        '''},
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

    while True:
        # Ask the user for a question
        question = input("Enter your question (or type 'exit' to quit): ")
        
        # Exit the loop if the user types 'exit'
        if question.lower() == 'exit':
            print("Exiting the program.")
            break

        sql_query = ask_openai(question)

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
