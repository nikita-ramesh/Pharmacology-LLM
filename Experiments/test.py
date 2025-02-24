import openai
import json
import requests
import time

openai.api_key = "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"

# Define your prompt-completion pairs directly within the file
examples = [
    {
        "prompt": "Find enzymes with endogenous substrates that have no pharmacology to speak of?",
        "completion": "select o.object_id, o.name from object o where o.object_id in (select distinct object_id from enzyme) and o.object_id in (select distinct object_id from endo_ligand_pairings) and o.object_id not in (select distinct object_id from interaction where object_id is not null and ((affinity_units != '-' and affinity_units is not null) or (original_affinity_units is not null and original_affinity_units != '-') or (concentration_range is not null and concentration_range != '-' and concentration_range != ''))); END"
    },
    {
        "prompt": "Write an essay on endothelian receptors antagonists",
        "completion": "select f.family_id, f.name, g.overview, g.comments, i.text from family f left join grac_family_text g on g.family_id=f.family_id left join introduction i on i.family_id=f.family_id where f.name ilike ('%endothelin%'); END"
    }
]

# Write examples to a JSONL file for fine-tuning
with open("fine_tuning_data.jsonl", "w") as f:
    for example in examples:
        json.dump(example, f)
        f.write("\n")

# Upload the file to OpenAI for fine-tuning
response = openai.File.create(
    file=open("fine_tuning_data.jsonl"),
    purpose="fine-tune"
)

# Get the file ID to use for fine-tuning
file_id = response['id']
print(f"File uploaded successfully with ID: {file_id}")

# Use requests.get to check file status
file_status_url = f"https://api.openai.com/v1/files/{file_id}"
file_status_response = requests.get(file_status_url, headers={'Authorization': f'Bearer {openai.api_key}'})
file_status = file_status_response.json()
print(f"File status: {file_status}")

# Start fine-tuning the model
fine_tune_response = openai.FineTune.create(
    training_file=file_id,
    model="curie"
)
print(f"Fine-tuning started: {fine_tune_response}")

# Wait for fine-tuning to complete before using the model
fine_tune_id = fine_tune_response['id']

while True:
    fine_tune_status = openai.FineTune.retrieve(fine_tune_id)
    if fine_tune_status['status'] == 'succeeded':
        print(f"Fine-tuning completed. Model ID: {fine_tune_status['fine_tuned_model']}")
        break
    elif fine_tune_status['status'] == 'failed':
        print("Fine-tuning failed.")
        break
    else:
        print("Fine-tuning is still in progress...")
        time.sleep(30)  # Wait for 30 seconds before checking the status again

# Once fine-tuning is completed, use the fine-tuned model for queries
while True:
    user_query = input("Ask a question (or type 'exit' to quit): ")

    if user_query.lower() == 'exit':
        print("Exiting the program.")
        break

    # Get the response from the fine-tuned model
    response = openai.ChatCompletion.create(
        model=fine_tune_status['fine_tuned_model'],  # Use the fine-tuned model ID
        messages=[
            {"role": "system", "content": "You are an SQL query generator for the Guide to Pharmacology."},
            {"role": "user", "content": user_query}
        ]
    )

    assistant_response = response['choices'][0]['message']['content']
    print(f"Assistant: {assistant_response}")
