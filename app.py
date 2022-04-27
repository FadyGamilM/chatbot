from flask import Flask, request
import json
# import tensorflow


from keras.models import load_model
# from tensorflow.keras.models import load_model
# model = load_model('chatbotmodel.h5')
# from bow_chatbot_arabic import predict_class,get_response,intents
import pickle
import nltk
import re
import numpy as np
import random
import json

app = Flask(__name__)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def cleanUP_sentence(sentence):
    sentence = re.sub(r"أ|آ|إ", "ا", sentence)
    sentence = re.sub(r"ه", "ة", sentence)
    sentence_Words = nltk.word_tokenize(sentence)
    return sentence_Words
  

def bag_of_words(sentence):
    sentence_words = cleanUP_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    error_thr = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_thr]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result=random.choice(i['responses'])
            break
    return result

@app.route('/chat', methods = ['POST'])
def chat():
   file = open("testfranco.json", encoding="utf8") 
   intents = json.load(file)
   msg = json.loads(request.data)
   print("given ", msg['msg'])
   ints = predict_class(msg['msg'])
   res = get_response(ints, intents)
   print("result", res)
   return {"response": res}

if __name__ == '__main__':
   app.run(debug=True)