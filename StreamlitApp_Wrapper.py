import faiss
import numpy as np
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer

def load_college_database(database_file):
    with open(database_file, 'r') as f:
        data = f.readlines()
    return data

def create_faiss_index(text_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_data, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, model

def retrieve_similar_text(query, index, model, text_data, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [text_data[i] for i in indices[0]]

def retrieve_similar_text_research(query, index, model, text_data, top_k=10):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [text_data[i] for i in indices[0]]

# Load the tokenizer and model
path = 'email_classifier_5'
model = AutoModelForSequenceClassification.from_pretrained(path)
label_map = {0: "General", 1: "Sensitive", 2: "Research"}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'})

# Streamlit app
st.title("RAG Email Reply Generator")

# Input text
email_txt = st.text_area("Enter the email text:")

if st.button("Generate Reply"):
    if email_txt:
        encoding = tokenizer(email_txt, return_tensors='pt', padding=True, truncation=True)
        output = model(**encoding)
        scores = output.logits
        # print(f"Scores: {scores}")
        st.write(f"Label of Mail Text: {label_map[scores.argmax().item()]}")
        label = label_map[scores.argmax().item()]

        if label == "General":
            # Generate a response with rag on the college database
            college_data = load_college_database("dataset/college_data.txt")
            index, faiss_model = create_faiss_index(college_data)
            similar_texts = retrieve_similar_text(email_txt, index, faiss_model, college_data)
            # print(similar_texts)
            augmented_prompt ="Example1: What is your post?\n Answer: I am the HOD of the department.\nExample2: Where can I get a LOR?\n Answer: Letter of recommendation can be requested from a faculty member.\n All the answers are from the related college data. Now answer this:\n" + email_txt + "\nRelated info from college database:\n" + "\n".join(similar_texts) + "\nAnswer Using the above information only." 

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model = GPT2LMHeadModel.from_pretrained("gpt2-medium", max_length=1024)
            model = GPT2LMHeadModel.from_pretrained("gpt2_finetuned", max_length=1024)
            
            # Tokenize the input and create attention mask
            inputs = tokenizer(augmented_prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Generate response
            response_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=20,  # Adjust as needed
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1
            )
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Remove the input prompt from the response
            generated_text = response[len(augmented_prompt):].strip()
            st.write(f"Response: {generated_text}")

        if label == "Research":
            # Generate a response with rag on the college database
            research_data = load_college_database("dataset/research_data.txt")
            index, faiss_model = create_faiss_index(research_data)
            similar_texts = retrieve_similar_text_research(email_txt, index, faiss_model, research_data)
            # print(similar_texts)
            augmented_prompt ="Example1: What is your post?\n Answer: I am the HOD of the department.\nExample2: Where can I get a LOR?\n Answer: Letter of recommendation can be requested from a faculty member.\n All the answers are from the related college data. Now answer this:\n" + email_txt + "\nRelated info from college database:\n" + "\n".join(similar_texts) + "\nAnswer Using the above information only." 

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model = GPT2LMHeadModel.from_pretrained("gpt2_finetuned_research", max_length=1024)
            
            # Tokenize the input and create attention mask
            inputs = tokenizer(augmented_prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Generate response
            response_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=30,  # Adjust as needed
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1
            )
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Remove the input prompt from the response
            generated_text = response[len(augmented_prompt):].strip()
            st.write(f"Response: {generated_text}")

        if label == "Sensitive":
            st.write("This is a sensitive email. Please handle it with care.")
            st.write("Sent the mail to the respective department for further action.")
    else:
        st.write("Please enter some text.")