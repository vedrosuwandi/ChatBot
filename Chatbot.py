import nltk
import tensorflow as tf
import json
import numpy as np
import pickle
import random
import tkinter as tk
from tkinter import *

# Load Model
model = tf.keras.models.load_model('Chatbot.h5')

lemmatizer = nltk.stem.WordNetLemmatizer()

intent = json.loads(open('intents.json').read())
tags = pickle.load(open('tags.pkl', 'rb'))


# Clean Sentence 
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, showDetails=True):
    # Open the word collection from the dataset
    words_collection = pickle.load(open('words.pkl', 'rb'))

    count = 0
    words = []

    # Clear the Sentence (lower case and Lemmatize)
    sentence_split = clean_sentence(sentence)

    # Initialize Token
    tokens = np.zeros(shape=(len(words_collection)), dtype=np.float32)
    
    # Check if there is a word found on the Word collection
    for word in sentence_split:
        if word in words_collection:
            tokens[words_collection.index(word)] = 1
            words.append(word)
            count += 1
    
    if showDetails:
        print(f"Found : {count} word(s) on the sentence")
        print(words)
    
    return tokens

def predict(model, sentence):
    inputs = bag_of_words(sentence, showDetails=False)
    inputs = tf.expand_dims(inputs, axis=0)

    pred = model.predict(inputs)

    pred = tf.squeeze(pred)

    # results = [[i, r.numpy()] for i, r in enumerate(pred) if r > 0.25]

    # results.sort(key=lambda x: x[1], reverse=True)

    # results_list = []
    # for result in results:
    #     results_list.append({"Intent" : tags[result[0]], "Probability" : str(f"{result[1]*100:.2f}%")})

    result = tf.math.argmax(pred)
    
    intent = tags[result]

    # print(f"Intent : {intent} - {tf.reduce_max(pred)*100:.2f}%", )

    return intent, tf.reduce_max(pred)*100
    
    
# Get Response
def get_response(model, sentence):
    res, prob = predict(model, sentence)

    intents_json = intent['intents']
    for intent_json in intents_json:
        if intent_json['tag'] == res:
            response = random.choice(intent_json['responses'])

    return response

    # print(intents_json)
    
    
    

# GUI
root = Tk()
root.title("Chatbot")

root.geometry("400x450")

def send():
    message = entrybox.get("1.0", 'end-1c').strip()
    entrybox.delete("0.0", END)
    if message != '':
        chatbox.config(state=NORMAL)
        chatbox.insert(END, f"You : {message} \n\n")
        chatbox.config(foreground='#446665', font=("Verdana", 12))


        response = get_response(model, message)

        chatbox.insert(END, f"Bot : {response} \n\n")
        chatbox.config(state=DISABLED)
        chatbox.yview(END)

        
# Create Chat Window
chatbox = Text(root, bd=0, bg='white', height=8, width=50, font='Arial')

chatbox.config(state=DISABLED)

# Bind Scrollbar
scrollbar = Scrollbar(root, command=chatbox.yview, cursor='heart')

chatbox['yscrollcommand'] = scrollbar.set

#Send Button
sendbutton = Button(root, font=("Verdana", 12, 'bold'), text="Send" , width="12", height=5, bd=0, bg = '#f9a602', activebackground='#3c9d9b', fg='#000000', command=send)

#Create the entry box
entrybox = Text(root, bd=0, bg='white', width='29', height='5', font="Arial")

# Place all components
scrollbar.place(x=376, y=6, height=386)
chatbox.place(x=6, y=6, height=386, width=370)
entrybox.place(x=128, y=401, height=30, width=265)
sendbutton.place(x=6, y=401, height=30)

root.mainloop()
