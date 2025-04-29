# **Pharmacology-LLM**  
Tuning Large Language Models (LLMs) for Text-to-SQL on the IUPHAR/BPS Guide to Pharmacology  

## **Overview**  
This project develops a **Text-to-SQL pipeline** for the **IUPHAR/BPS Guide to Pharmacology**, enabling researchers to query the database using natural language. The system leverages **GPT-4o**, which has been **prompt-engineered** for improved SQL generation.  

### **Pipeline Workflow**  
1. **User Query Input** – The user provides a natural language query (NLQ).  
2. **LLM Processing** – GPT-4o converts the NLQ into a structured SQL query using a carefully designed prompt.  
3. **SQL Execution** – The generated SQL query is executed on the **Guide to Pharmacology** PostgreSQL database.  
4. **Results Retrieval** – The system returns relevant results to the user.  

### **Prompt Engineering Components**  
The prompt consists of four key elements:  
- **Database Schema** – A structured representation of the database sent to the LLM.  
- **Many-shot Learning (51-shot)** – Examples from the custom **NLQ-SQL development dataset** to improve SQL generation accuracy.  
- **Manual Refinement Rules** – Manually predefined rules to enhance query accuracy and reduce errors.  
- **Self-Correction for Error Handling** – Adjustments to mitigate invalid SQL outputs.  

---

## **Project Structure**  

 
```
Pharmacology-LLM
├───Experiments                 # Older experiments and preliminary tests 
├───Results                     # Intermediate visualisations and graphs from development 
├───Training                    # NLQ-SQL dataset (development + test set) 
└───Visualise                   # Final visualisations and results 
schema_structure.json           # Database schema mapping table names to column names 
implementation_on_split.py      # Runs the pipeline on a 50-50 split of the dataset 
implementation_on_test.py       # Runs the pipeline on the final test set 
query_results.txt               # Results from the test set execution 
split_results.txt               # Results from the development set execution
```

---

## **Running the Pipeline**  

### **Prerequisites**  
1. **OpenAI API Key**  
   - Set your API key as an environment variable:  
     ```sh
     set OPENAI_API_KEY="your-api-key-here"
     ```  
2. **Guide to Pharmacology Database**  
   - The PostgreSQL database must be stored locally with the following configurations:  
     ```python
     'host': 'localhost',
     'database': 'guide_to_pharmacology',
     'user': 'postgres',
     'password': pwd()  # You will be prompted to enter your password at runtime.
     ```
   - **DBeaver** can be used to set up the local database.  

### **Execution Commands**  
To run the pipeline, execute:  
- **To run on the development dataset (50-50 split):**  
  ```sh
  python implementation_on_split.py
  ```
- **To run on the final test dataset:**
    ```sh
    python implementation_on_test.py
    ```
### **Output Files**
Running these scripts generates the following result files:

**query_results.txt** – Contains the final results for the test set.

**split_results.txt** – Contains results for the development set.

These files are included in the repository and were used in the dissertation analysis.
