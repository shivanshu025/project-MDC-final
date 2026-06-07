from sentence_transformers import SentenceTransformer

print("Downloading and saving model locally...")
# This fetches the model from the internet one last time
model = SentenceTransformer('all-MiniLM-L6-v2')

# This creates a folder named 'local_model' right next to your script
model.save('./local_model')
print("Done! The model is saved in the './local_model' folder.")