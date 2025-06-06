from tools import remove_punctuations, stemming, all_words

train = []
target = []

def create_vector(text, words_list):
    vec = []
    puncRemoved = remove_punctuations(text)
    textAsList = puncRemoved.lower().split()
    stemmedList = stemming(textAsList)
    for w in words_list:
        vec.append(1 if w in stemmedList else 0)
    return vec

def create_data(dt, words_list, intents_mapping):
    for d in dt["data"]:
        inte = d["intent"]
        queries = d["query"]
        for q in queries:
            vector = create_vector(q, words_list)
            train.append(vector)
            target.append(intents_mapping[inte])
    return train, target


