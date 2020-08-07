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
        score = self.sentiment_calculator(self.tokenizer(text), self.posDict, self.negDict) * 1000
        if score > 0:
            sentiment = "Positive: %.4f" % score + "%"
        elif score == 0:
            sentiment = "Netural: %.4f" % score + "%"
        else:
            sentiment = "Negative: %.4f" % score + "%"
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
              2 : 'Growth Outlook', 
              3 : 'Financial Market',
              4 : 'Labor Market', 
              5 : 'Investment'
        }
    
    
    def predict(self, text):
        topic_weights = self.model.transform([self.tokenizer(text)])[0]
        topic_weights_percentage = [str(round(100*weight, 2)) + "%" for weight in list(topic_weights)]
        return str(dict(zip(self.topic_dict.values(), topic_weights_percentage)))
    
topicAnalyzer = TopicAnalyzer(simple_tokenizer, lda_pipe)


tfidf_svc_pipe = pickle.load(open(file_path_prefix + 'models/tfidf_svc_pipe.pkl', 'rb'))
class SlpoeAnalyzer:
    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer
        self.class_dict = {
              0 : 'Flatten',
              1 : 'Steepen'
        }
    
    
    def predict(self, text):
        class_prob = self.model.predict_proba([self.tokenizer(text)])[0]
        class_prob_percentage = [str(round(100*weight, 2)) + "%" for weight in list(class_prob)]
        return str(dict(zip(self.class_dict.values(), class_prob_percentage)))

slopeAnalyzer = SlpoeAnalyzer(simple_tokenizer, tfidf_svc_pipe)


app = FastAPI()

@app.get("/predict/{minutes_paragraph}")
async def predict_minutes_paragraph(minutes_paragraph: str):
    return ({
        "Topic": topicAnalyzer.predict(minutes_paragraph),
        "Sentiment": sentimentAnalyzer.predict(minutes_paragraph),
        "Slope": slopeAnalyzer.predict(minutes_paragraph)
    })


