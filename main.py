from fastapi import FastAPI
import dill as pickle
from spacy.lang.en import English

en = English()


file_path_prefix = '/Users/aaroncgw/Google Drive/fednlp/'

with open(file_path_prefix + 'data/sentiment/sentiment_pos_dict.pickle', 'rb') as handle:
    posDict = pickle.load(handle)
    
with open(file_path_prefix + 'data/sentiment/sentiment_neg_dict.pickle', 'rb') as handle:
    negDict = pickle.load(handle)

def simple_tokenizer(doc, model=en):
    tokenized_docs = []
    parsed = model(doc)
    return([t.lemma_.lower() for t in parsed if (t.is_alpha)&(not t.like_url)&(not t.is_stop)])


def RetrieveScore(tokenized_para, posDict, negDict): 
    pos_sum = 0
    neg_sum = 0
    score = 0
    if len(tokenized_para) <8:
        return 0
    for word in tokenized_para: 
        if word in posDict:
            pos_sum +=1
        elif word in negDict:
            neg_sum +=1
    try:
        score = ((pos_sum-neg_sum)/(pos_sum+neg_sum))*(1/len(tokenized_para)) #should this be 
    except ZeroDivisionError:
        score = 0
    return score


class SentimentAnalyzer:
    def __init__(self, tokenizer, sentiment_calculator, posDict, negDict):
        self.tokenizer = tokenizer
        self.sentiment_calculator = sentiment_calculator
        self.posDict = posDict
        self.negDict = negDict
    
    
    def predict(self, text):
        sentiment = None
        score = self.sentiment_calculator(self.tokenizer(text), self.posDict, self.negDict)
        if score > 0:
            sentiment = "Positive: %.4f" % score
        else:
            sentiment = "Negative: %.4f" % score
        return sentiment

sentimentAnalyzer = SentimentAnalyzer(simple_tokenizer, RetrieveScore, posDict, negDict)




lda_pipe = pickle.load(open(file_path_prefix + 'models/lda_pipe.pkl', 'rb'))
class TopicAnalyzer:
    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer
        self.topic_dict = {
                  0 : 'Inflation',
                  1 : 'Economic Policy',
                  2 : 'Investment', 
                  3 : 'Financial Market',
                  4 : 'Labor Market', 
                  5 : 'Growth Outlook'
             }
    
    
    def predict(self, text):
        topic_weights = self.model.transform([self.tokenizer(text)])[0]
        topic_weights_percentage = [str(100 * round(weight,4)) + "%" for weight in list(topic_weights)]
        return str(dict(zip(self.topic_dict.values(), topic_weights_percentage)))
    
topicAnalyzer = TopicAnalyzer(simple_tokenizer, lda_pipe)


model_pipe = pickle.load(open(file_path_prefix + 'models/bert_svc.pkl', 'rb'))



app = FastAPI()

@app.get("/predict/{minutes_paragraph}")
async def predict_minutes_paragraph(minutes_paragraph: str):
    return ({
        "Topic": topicAnalyzer.predict(minutes_paragraph),
        "Sentiment": sentimentAnalyzer.predict(minutes_paragraph)
        #"Prediction": model_pipe.predict(text)
    })


