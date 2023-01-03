# import the streamlit library
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import pandas as pd
import torch 
# from tensorflow.keras.saving import pickle_utils

import matplotlib
import matplotlib.pyplot as plt

#from summarizer import TransformerSummarizer

from transformers import pipeline 
summarizer = pipeline("summarization")

from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# st.write(st.__version__)

df2 = pd.read_csv('D:/Megh/megh/Sem 7/Electives/ANN/Project/miyank/file1.csv', usecols=['Headline', 'Content'])
  
#from_i = 10
#count = 5
headlines = df2['Headline']
headlines = headlines.to_list()
df2 = df2['Content']
df2 = pd.DataFrame(df2)
df2.reset_index(inplace=True, drop=True)

st.dataframe(df2)

# Creation the list with new long block
max_length = 37  # minimum characters in each block
i = 0
bodies = []
while i < len(df2):
    body = ""
    body_empty = True
    while (len(body) < max_length) and (i < len(df2)):
        if body_empty:
            body = df2.loc[i,'Content']
            body_empty = False
        else: body += " " + df2.loc[i,'Content']
        i += 1
    bodies.append(body)
    st.write("Length of blocks =", len(body))
# st.write(f"\nNumber of text blocks = {len(bodies)}\n")
# st.write("Text blocks:\n", bodies)

min_length_text = 60

# bert_summary = []
# for i in range(len(bodies)):
#     bert_model = Summarizer()
#     bert_summary.append(''.join(bert_model(bodies[i], min_length=min_length_text)))

# gpt_summary = []
# for i in range(len(bodies)):
#     GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
#     gpt_summary.append(''.join(GPT2_model(bodies[i], min_length=min_length_text)))

gpt_summary = []
for i in range(len(bodies)):
    #GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    gpt_summary.append(''.join(summarizer(bodies[i], min_length=min_length_text)))

st.write("All Summarizing Results:\n")
# for i in range(len(bodies)):
#     st.write("ORIGINAL TEXT:")
#     st.write(bodies[i])
#     st.write("\nBERT Summarizing Result:")
#     st.write(bert_summary[i])
#     st.write("\nOriginal headline:")
#     st.write(headlines[i])
#     st.write("\n\n")

st.write("ORIGINAL TEXT:")
st.write(bodies[i])
st.write("\nBERT Summarizing Result:")
#st.write(bert_summary[0])
st.write(gpt_summary[i])
