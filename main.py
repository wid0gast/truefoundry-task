from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import spacy

app = FastAPI()

tokenizer = spacy.load('en_core_web_md')
loaded_model = pickle.load(open('svm_rbf_1_scaler.pkl', 'rb'))

class Tweet(BaseModel):
    text: str

@app.post('/predict')
async def predict_sentiment(tweet: Tweet):
    data = tweet.dict()
    print(data['text'])
    vect = tokenizer(data['text']).vector
    pred = loaded_model.predict(vect.reshape(1,-1))

    if pred == 0:
        return 'negative'
    else:
        return 'positive'