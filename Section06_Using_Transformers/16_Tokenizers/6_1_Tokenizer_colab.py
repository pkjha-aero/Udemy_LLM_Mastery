#!/usr/bin/env python
# coding: utf-8

# # 🤖 **Understanding Tokenizers with BERT**
# 
# This notebook shows how to use BERT tokenizers to turn text into data that the model can understand.
# 
# ## 🛠️ Setup and Installation
# 
# First, we need to install the libraries we will use.

# In[1]:


get_ipython().system('pip install pandas==2.0.1')
get_ipython().system('pip install transformers==4.29.2')


# 
# ## 📚 Importing Libraries
# 
# We import the libraries necessary for our tasks.

# In[2]:


# Import required libraries
from transformers import BertModel, AutoTokenizer
import pandas as pd


# 
# ## 🤖 Model Setup
# 
# We load a pre-trained BERT model and its tokenizer.

# In[3]:


# Specify the pre-trained model to use: BERT-base-cased
model_name = "bert-base-cased"


# In[4]:


# Instantiate the model and tokenizer for the specified pre-trained model
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer


# 
# ## 📝 Tokenizing Text
# 
# We use the tokenizer to turn a sentence into tokens.

# In[5]:


# Set a sentence for analysis
sentence = "When life gives you lemons, don't make lemonade."


# In[6]:


# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
tokens


# 
# ## 📘 Vocabulary and Token IDs
# 
# We create a DataFrame to see the tokenizer's vocabulary and sort it by token IDs.

# In[7]:


# Create a DataFrame with the tokenizer's vocabulary
vocab = tokenizer.vocab
vocab_df = pd.DataFrame({"token": vocab.keys(), "token_id": vocab.values()})
vocab_df = vocab_df.sort_values(by="token_id").set_index("token_id")

vocab_df


# ## 🔍 Encoding and Decoding
# 
# Encode the sentence into IDs and then decode it back to text.

# In[8]:


# Encode the sentence into token_ids using the tokenizer
token_ids = tokenizer.encode(sentence)
token_ids


# 
# ## 🔎 Compare Token Lengths
# 
# Compare the length of tokens and token IDs.
# 

# In[9]:



# Print the length of tokens and token_ids
print("Number of tokens:", len(tokens))
print("Number of token IDs:", len(token_ids))


# 
# ## 🔄 Explore Token Data
# 
# Look at specific tokens by their IDs.

# In[10]:


# Access the tokens in the vocabulary DataFrame by index
print("Token at position 101:", vocab_df.iloc[101])
print("Token at position 102:", vocab_df.iloc[102])


# ## 📃 Token and ID Pairing
# 
# Show pairs of tokens and their IDs.

# In[11]:


# Zip tokens and token_ids (excluding the first and last token_ids for [CLS] and [SEP])
list(zip(tokens, token_ids[1:-1]))


# In[12]:


# Decode the token_ids (excluding the first and last token_ids for [CLS] and [SEP]) back into the original sentence
tokenizer.decode(token_ids[1:-1])


# In[13]:


# Tokenize the sentence using the tokenizer's `__call__` method
tokenizer_out = tokenizer(sentence)
tokenizer_out


# 
# ## 🧩 Handling Multiple Sentences
# 
# Tokenize two sentences with and without padding, and decode them.

# In[14]:


# Create a new sentence by removing "don't " from the original sentence
sentence2 = sentence.replace("don't ", "")
sentence2


# In[15]:


# Tokenize both sentences with padding
tokenizer_out2 = tokenizer([sentence, sentence2], padding=True)
tokenizer_out2


# In[16]:


# Decode the tokenized input_ids for both sentences
tokenizer.decode(tokenizer_out2["input_ids"][0])


# In[17]:


tokenizer.decode(tokenizer_out2["input_ids"][1])


# 
# ## 🌟 Conclusion
# 
# This notebook walked you through how to use a BERT tokenizer to process text, turning it into tokens and IDs, and how to handle multiple sentences. Feel free to change the sentences or explore more functions of the tokenizer.
