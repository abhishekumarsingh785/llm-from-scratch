# Shakespearean Text Generator

Welcome to the **Shakespearean Text Generator**! This project combines the elegance of Shakespearean prose with the power of modern deep learning. Trained from scratch on the Bard's complete works, this custom language model generates text in Shakespeare's timeless style.

## 🎯 Project Highlights

### Custom GPT-2 Inspired Architecture:
- Developed from scratch using PyTorch.
- Trained on the complete works of Shakespeare to capture the essence of his writing.

### Interactive Streamlit App:
- Input any phrase, and get a Shakespearean continuation.
- Designed with an intuitive interface for ease of use.

### Learning Milestone:
- **Understanding Transformer Models**: Learned to implement multi-head self-attention, positional embeddings, and feed-forward layers.
- **Model Training**: Mastered techniques like dataset preparation, tokenization, and training loop optimization.
- **Fine-Tuning**: Experimented with hyperparameters, dropout rates, and optimizer configurations to enhance performance.

## 📽 Demo

Experience the app in action by watching the demo video:  
[Watch the Demo](https://drive.google.com/file/d/1dOQchpZTvxDPiRXMvTSnCtfAReHQMyUh/view?usp=sharing)

## ⚙️ Technical Specifications

### Model Configuration:
- Vocabulary size: **50,257 tokens**
- Context length: **256 tokens**
- Embedding dimension: **768**
- Attention heads: **12**
- Transformer layers: **12**
- Dropout rate: **10%**

### Training Setup:
- Dataset: Complete works of Shakespeare (tokenized with GPT-2 tokenizer).
- Optimizer: AdamW with a learning rate of \(5 \times 10^{-4}\).
- Epochs: **10**
- Loss Function: Cross-Entropy Loss.

## ✨ Sample Output

- **Input**: `"To be or not to be"`
- **Output**: `"To be or not to be, that is the noble question that doth ponder the hearts of men eternal."`

## 🔮 Future Directions

- Extend training to include other classical literature for broader stylistic capabilities.
- Add options for fine-tuning on user-provided datasets.
- Introduce multilingual Shakespearean text generation.

## 📘 Interactive Learning with Jupyter Notebook
LLM_from_Scratch.ipynb
To dive deeper into the implementation and understand how each block of code works, check out the accompanying Jupyter Notebook:  
[LLM_from_Scratch.ipynb](LLM_from_Scratch.ipynb)

### What You'll Learn:
- How multi-head attention processes text input.
- The role of positional embeddings and feed-forward layers in the transformer model.
- How the training loop is structured, including dataset preparation and loss calculation.
- Generate Shakespearean text step-by-step and analyze model outputs.

### Perfect For:
- **Beginners**: Learn how transformers and language models work.
- **Practitioners**: Explore code implementation and experiment with your changes.

## 🛠 Files in the Repository

- **LLM_from_scratch.py**: Training script for the custom GPT-2 inspired model.
- **app.py**: Code for the Streamlit-based text generation app.
- **LLM_from_Scratch.ipynb**: An interactive notebook that guides you through the architecture, training, and functionality of a custom large language model.

