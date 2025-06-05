import re #regular expressions
from nltk.stem import PorterStemmer # need to install nltk in local env
# I used the command: sudo apt install python3-nltk

ps  = PorterStemmer() #instantiating imported stemmer
words = []

# removing punctuations
def remove_punctuations(text):
    return re.sub(r'[^\w\s]', "", text)

def stemming(word_list):
    return [ps.stem(w) for w in word_list]
    # not using lower() here and instead in the calling function

def all_words(dt):
    for d in dt["data"]:
        for q in d["query"]:
            punc_removed = remove_punctuations(q)
            stemmed_list = stemming(punc_removed.lower().split())
            words.extend(stemmed_list)
    return list(set(words))