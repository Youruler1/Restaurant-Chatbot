import json
import torch
import random
from tools import all_words
from create_samples import create_vector, create_data
from train_file import train_fn

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
# print(f"{len(words_list)} {len(train[0])}")
model = train_fn(train, target, len(words_list), len(dt["data"]))
# passing into the train_fn training and target data, and the dimensions of the vocab and no. of intents


# print(len(dt["data"][pred_num]["responses"]))

# chatting 
while True:
    inp = input("You: ")
    vec = create_vector(inp, words_list)
    input_vec = torch.as_tensor(vec, dtype=torch.float32)[None, ...]
    output = model(input_vec)
    pred_num = output.argmax().item()
    resps = intents_response[pred_num]
    # print("Bot: ", resps[random.randint(0, len(dt["data"][pred_num]["responses"]) - 1)])
    print("Bot: ", resps[random.randint(0, 2)])





# parsed and loaded the json file as a list into a variable dt
# created vocabulary all_words (using fns in tools.py) from all the queries (cleaned) in dt
# created numerical mappings of intents and responses
# made train and target data using cleaning and vectorization
# wrote script to define model in pytorch, and train it with prepared data
