#!/usr/bin/env python
# coding: utf-8

# ### Import Libaries

# In[1]:


import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

##torch and embedding libaries
from transformers import AutoTokenizer, AutoModel
import torch

import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


# #### Load csv

# In[3]:


trans_df = pd.read_csv('transactions.csv')
user_df = pd.read_csv('users.csv')


# #### Clean description column

# In[4]:


# Clean description column
def clean_data(df):
    """_summary_

    Args:
        df (DataFrame): Pandas dataframe with of transactions made by users .
    Returns:
        df: returns a clean version of the dataframe
    """
    df['description'] = df['description'].str.strip().str.lower()#.str.replace('[^\w\s]', '')
    return df


# ### Task 1 Function

# In[5]:


def calculate_match_metric(df, name):
    """_summary_

    Args:
        df (DataFrame): Pandas dataframe with of transactions made by users .
        name (str): name of users
    Returns:
        dict: dict contains user's id and matched metrics ranked based on metric_score
    """

    # Prepare response data
    response_data = {}
    matched_transactions = []

    # Calculate match metric for each transaction
    for index, transaction in df.iterrows():
        name= name
        transaction_id = transaction['id']
        transaction_description = transaction['description']

        # Initialize match metric
        match_metric = 0

        # Perform fuzzy matching to calculate match metric
        match_metric = fuzz.token_sort_ratio(name, transaction_description)
        # Create a dictionary for the matched transaction
        matched_transaction = {
            'id': transaction_id,
            'match_metric': match_metric
        }

        # Add the matched transaction to the list
        matched_transactions.append(matched_transaction)

    # Sort the matched transactions in descending order of match metric
    matched_transactions = sorted(matched_transactions, key=lambda x: x['match_metric'], reverse=True)
    # Add matched transactions and total number of matches to the response data
    response_data['transactions'] = matched_transactions
    response_data['total_number_of_matches'] = len(matched_transactions)
    return response_data
    


# #### Considered Edge Cases for Task 1

# Here's a list of some possible variations we might encounter:
# 
# 1. `name` is in the users dataframe/table.
# 2. `name` is not of not of type `str`.
# 3. when a dataframe is empty.
# 
# 
# Edge cases not taken care of
# 
# 1. `name` is not in the users dataframe/table.
# 2. when we want to search for `multiple name` is the transaction table.
# 

# ### Task 2 Function
# 
# Calculate embeddings for  all transactions that has similar descriptions to the input sting

# In[9]:


# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def find_similar_transactions(df, input_text:str):
    # Tokenize input text
    input_tokens = tokenizer.tokenize(input_text)
    total_tokens_used = len(input_tokens)

    # Convert tokens to IDs and create input tensors
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_tensor = torch.tensor([input_ids])

    # Get embeddings for input text
    with torch.no_grad():
        output = model(input_tensor)

    input_embedding = output[0].mean(dim=1).squeeze()

    # Find similar transactions
    similar_transactions = []
    for _, transaction in df.iterrows():
        description = transaction['description']
        transaction_tokens = tokenizer.tokenize(description)
        transaction_ids = tokenizer.convert_tokens_to_ids(transaction_tokens)
        transaction_tensor = torch.tensor([transaction_ids])

        with torch.no_grad():
            output = model(transaction_tensor)

        transaction_embedding = output[0].mean(dim=1).squeeze()
        similarity = torch.cosine_similarity(input_embedding, transaction_embedding, dim=0).item()

        similar_transactions.append({'id': transaction['id'], 'embedding': transaction_embedding.tolist(), 'total_number_of_tokens_used': total_tokens_used, 'similarity': similarity})

    # Sort transactions by similarity in descending order
    similar_transactions.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_transactions



# ### API end point for task 1 and Task 2

# In[10]:


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.encoders import jsonable_encoder
import nest_asyncio
import uvicorn

app = FastAPI()

class User(BaseModel):
    name: Optional[str] = None
        
class Description(BaseModel):
    text: Optional[str] = None
        
@app.get("/")
def root():
    return {"message": "EndPoint is working perfectly!"}

@app.post('/match_transactions')
def match_transactions(name:User):

    df = clean_data(trans_df)
    response_data = calculate_match_metric(df, name)
    #print(json.dumps(response_data))
    # Return the response as JSON
    return jsonable_encoder(response_data)


        
@app.post('/find_similar_transactions')
def match_transactions_endpoint(text:Description):
    df = clean_data(trans_df)
    response_data = find_similar_transactions(df, text.text)
#     print(response_data)
    # Return the response as JSON
    return jsonable_encoder(response_data)



# In[ ]:


print('Navigate to this endpoint to test app: http://127.0.0.1:8000/docs#/default/')
nest_asyncio.apply()

uvicorn.run(app, host="0.0.0.0", port=8000)


# ### Task 3

# Given additional resources, suggest (in the README/pdf/keynote or otherwise), how you might take this proof of concept to production. Include any changes or improvements you might make.

# #### Answer

# Here are some suggestions for taking this proof of concept to production:
# 
# **Improvements to current implementation:**
# 
# 1. Add more robust input validation, sanitization, error handling
# 2. Containerize with Docker for easy deployment
# 3. Add API keys, rate limiting, authentication
# 4. Improve exception handling and logging
# 5. Add automated tests (unit, integration)
# 6. Enable CORS if front-end consuming the API is on different domain
# 7. User a more efficient embedding function like Faiss or OpenAIEmbeddings
# 8. Improve the run ime of the script
# 
# 
# **Scaling:**
# 
# 1. Switch to a production-grade database like PostgreSQL to store transaction/user data
# 2. Add caching layer (Redis) to reduce load on database for common queries
# 3. Optimize DB queries, add indexes for performance
# 4. User Large Language Models and vector database to improve query search.
# 
# 
# **Monitoring:**
# 
# 1. Add performance monitoring (Prometheus) to track API latency, error rates
# 2. Track usage metrics on all endpoints
# 3. Set up logging aggregation (Elasticsearch) for analyzing logs
# 
# **Model serving:**
# 
# 1. Serve sentence embedding model on dedicated model server (Seldon Core)
# 2. Implement A/B testing for model variations
# 3. Build pipeline for retraining model on new data
# 

# In[ ]:





# In[ ]:


get_ipython().system('pip install fuzzywuzzy')
# !pip install python-Levenshtein

