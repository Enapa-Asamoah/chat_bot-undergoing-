import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import process
import tensorflow as tf

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list):
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def match_intent(sentence, intents):
    all_patterns = [pattern for intent in intents['intents'] for pattern in intent['patterns']]
    best_match = process.extractOne(sentence, all_patterns)
    return best_match

print("GO! Bot is running!")

while True:
    print("You: ")
    message = input("")
    match = match_intent(message, intents)
    if match and match[1] > 80:  # Match threshold
        matched_pattern = match[0]
        for intent in intents['intents']:
            if matched_pattern in intent['patterns']:
                ints = [{'intent': intent['tag'], 'probability': '1.0'}]
                break
        res = get_response(ints)
        print("Chat: ")
        print(res)
    else:
        print("Chat: I didn't understand that.")
