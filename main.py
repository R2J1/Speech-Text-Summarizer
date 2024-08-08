from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(".")
model = AutoModelForSeq2SeqLM.from_pretrained(".")

# Read input text from a file
input_file_path = "transcript.txt"
with open(input_file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

# Generate summary
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Write the generated summary to a text file
output_file_path = "summary.txt"
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(summary)

print("Summary has been written to:", output_file_path)
