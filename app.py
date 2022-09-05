from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import spacy
import uvicorn

app = FastAPI()

tokenizer = spacy.load('en_core_web_md')
loaded_model = pickle.load(open('svm_rbf_1_scaler.pkl', 'rb'))

class Tweet(BaseModel):
    text: str

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}

@app.post('/predict')
async def predict_sentiment(tweet: Tweet):
    data = tweet.dict()
    print(data['text'])
    vect = tokenizer(data['text']).vector
    pred = loaded_model.predict(vect.reshape(1,-1))

    if pred == 0:
        return {'sentiment' : 'negative'}
    else:
        return {'sentiment' : 'positive'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', debug=True)