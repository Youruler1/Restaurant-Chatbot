import json
from tools import all_words
from create_samples import create_data

f = open("all_intents_js.json")
dt = json.load(f)

#vocab
words_list = sorted(all_words(dt))
# print(words_list)

intents_mapping = {}
intents_response = {}
for i, d in enumerate(dt["data"]):
    intents_mapping[d["intent"]] = i
    intents_response[i] = d["responses"]
reverse_mapping = {intents_mapping[k] : k for k in intents_mapping.keys()} 
    
# creating training data
train, target = create_data(dt, words_list, intents_mapping)
print(len(train))