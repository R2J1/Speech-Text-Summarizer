import pandas as pd
import torch
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForSeq2SeqLM.from_pretrained("./")

# Load the test data from test.csv
test_data = pd.read_csv("test.csv")

# Initialize the Rouge object
rouge = Rouge()

# Function to generate summaries
def generate_summary(article):
    inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs.input_ids.to(torch.device('cpu')), 
                                 attention_mask=inputs.attention_mask.to(torch.device('cpu')),
                                 length_penalty=0.8,
                                 num_beams=8,
                                 max_length=128)
    decoded_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return decoded_summary

# Lists to store reference summaries and generated summaries
references = []
hypotheses = []

# Iterate over the first 10 rows in the test data and generate summaries
for index, row in test_data.head(10000).iterrows():
    article = row["article"]
    reference_summary = row["highlights"]
    generated_summary = generate_summary(article)
    
    references.append(reference_summary)
    hypotheses.append(generated_summary)
    
    print("Original Article:")
    print(article)
    print("\nReference Summary:")
    print(reference_summary)
    print("\nGenerated Summary:")
    print(generated_summary)
    print("----------------------------------------")

# Calculate ROUGE scores
rouge_scores = rouge.get_scores(hypotheses, references, avg=True)

# Extract ROUGE-1, ROUGE-2, and ROUGE-L scores
rouge_1_score = rouge_scores['rouge-1']['f']
rouge_2_score = rouge_scores['rouge-2']['f']
rouge_l_score = rouge_scores['rouge-l']['f']

print("\nROUGE-1 Score:", rouge_1_score)
print("ROUGE-2 Score:", rouge_2_score)
print("ROUGE-L Score:", rouge_l_score)

