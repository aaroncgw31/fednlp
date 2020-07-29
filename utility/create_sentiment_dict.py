#https://github.com/hanzhichao2000/pysentiment


import abc
import re
import nltk
import pickle
import pandas as pd

class BaseTokenizer(object, metaclass=abc.ABCMeta):
    """
    An abstract class for tokenize text.
    """

    @abc.abstractmethod
    def tokenize(self, text):
        """Return tokenized temrs.
        
        :type text: str
        
        :returns: list 
        """
        pass


class Tokenizer(BaseTokenizer):
    """
    The default tokenizer for ``pysentiment``, which only takes care of words made up of ``[a-z]+``.
    The output of the tokenizer is stemmed by ``nltk.PorterStemmer``. 
    
    The stoplist from https://www3.nd.edu/~mcdonald/Word_Lists.html is included in this
    tokenizer. Any word in the stoplist will be excluded from the output.
    """
    
    def __init__(self):
        self._stemmer = nltk.PorterStemmer()
        self._stopset = self.get_stopset()
        
    def tokenize(self, text):
        tokens = []
        for t in nltk.regexp_tokenize(text.lower(), '[a-z]+'):
            t = self._stemmer.stem(t)
            if not t in self._stopset:
                tokens.append(t)
        return tokens
        
    def get_stopset(self):
        files = ['Currencies.txt', 'DatesandNumbers.txt', 'Generic.txt', 'Geographic.txt', 'Names.txt']
        stopset = set()
        for f in files:
            fin = open('%s/%s'%('/Users/aaroncgw/PycharmProjects/fednlp/data/sentiment', f), 'rb')
            for line in fin.readlines():
                line = line.decode(encoding='latin-1')
                match = re.search('(\w+)', line)
                if match == None:
                    continue
                word = match.group(1)
                stopset.add(self._stemmer.stem(word.lower()))
            fin.close()
        return stopset

_tokenizer = Tokenizer()

def tokenize(text):
        """
        :type text: str
        :returns: list
        """

        return _tokenizer.tokenize(text)

def tokenize_first(x):
        """
        :type x: str
        :returns: str
        """
        tokens = tokenize(x)
        if tokens:
            return tokens[0]
        else:
            return None


if __name__ == '__main__':
    hiv = pd.read_csv('/Users/aaroncgw/PycharmProjects/fednlp/data/sentiment/HIV-4.csv',dtype ='category')
    lm = pd.read_csv('/Users/aaroncgw/PycharmProjects/fednlp/data/sentiment/LM.csv')
    
    HIVp_lower = set(hiv.query('Positiv == "Positiv"')['Entry'].apply(tokenize_first).dropna())
    HIVn_lower = set(hiv.query('Negativ == "Negativ"')['Entry'].apply(tokenize_first).dropna())

    LMp_lower = set(lm.query('Positive > 0')['Word'].apply(tokenize_first).dropna())
    LMn_lower = set(lm.query('Negative > 0')['Word'].apply(tokenize_first).dropna())

    posDict = set(HIVp_lower|LMp_lower)
    negDict = set(HIVn_lower|LMn_lower)
    
    with open('sentiment_pos_dict.pickle', 'wb') as handle:
        pickle.dump(posDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('sentiment_neg_dict.pickle', 'wb') as handle:
        pickle.dump(negDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
