import base64
import codecs
import requests
import psycopg2
import pandas as pd
import json
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

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

# Load schema structure and create vector store
def load_schema_and_create_vectorstore():
    # Load schema structure from JSON file
    with open('schema_structure.json', 'r') as schema_file:
        schema_structure = json.load(schema_file)

    # Prepare documents for vectorization
    documents = []
    for table in schema_structure['tables']:
        table_name = table['table_name']
        columns = ', '.join(table['columns'])
        content = f"Table: {table_name}\nColumns: {columns}"
        documents.append({"page_content": content, "metadata": {"table_name": table_name}})

    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embedding_function)
    return vectorstore

# Function to ask a question and get SQL using OpenAI API
def ask_openai(question, vectorstore):
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """You are an SQL expert. Given the following database schema:
        {context}
        
        Generate an SQL query to answer the following question:
        {question}
        
        Return only the SQL query, nothing else."""
    )
    
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "question": question})
    return response.content

# Function to execute SQL query
def execute_query(conn, query):
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

# Main function
def main():
    conn = connect_to_db()
    if not conn:
        return

    vectorstore = load_schema_and_create_vectorstore()

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        
        if question.lower() == 'exit':
            print("Exiting the program.")
            break

        sql_query = ask_openai(question, vectorstore)

        if sql_query:
            print(f"Generated SQL Query: {sql_query}")

            results = execute_query(conn, sql_query)
            if results is not None and not results.empty:
                print("Query Results:")
                print(results)
            else:
                print("No results found.")

    conn.close()

if __name__ == "__main__":
    main()