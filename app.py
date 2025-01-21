import torch
import tiktoken
import streamlit as st
from gpt_functions import GPTModel, generate_text_simple, token_ids_to_text, text_to_token_ids, generate  # Import required functions and classes

# Load the model (ensure this matches your saved model structure and path)
GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

model_path = 'model_10B.pth'
model = GPTModel(GPT_CONFIG_124M)  # Make sure GPT_CONFIG_124M is defined or imported
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Move model to GPU if available for faster processing
if torch.cuda.is_available():
    model.to('cuda')
tokenizer = tiktoken.get_encoding("gpt2")
st.title('Shakesperean Text Generation App')
user_input = st.text_area("Enter your prompt:","", placeholder="Type your text here...")
if st.button('Generate Text'):
    if user_input:
        # Convert text to token ids
        token_ids = text_to_token_ids(user_input, tokenizer).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate text
        #generated_token_ids = generate_text_simple(model, token_ids, max_new_tokens=50, context_size=GPT_CONFIG_124M["context_length"])
        generated_token_ids = generate(model, token_ids, max_new_tokens=50, context_size=GPT_CONFIG_124M["context_length"], top_k=25, temperature=1.2)
        # Convert tokens to text
        generated_text = token_ids_to_text(generated_token_ids, tokenizer)
        
        # Display the generated text
        st.write(generated_text)
    else:
        st.write("Please enter a prompt to generate text.")


