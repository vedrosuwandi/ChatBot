# ChatBot
Simple Chat Bot App using Neural Network Based on First Python Chatbot by Shivashish Thkaur 

![Github Repo stars](https://img.shields.io/badge/Python-v2.8-green)
![Github Repo stars](https://img.shields.io/badge/tensorflow-v2.9.1-blue)

This project is using a Text Classification to predict the intentions or the purpose of the conversation eg. Greetings, Goodbye, Thanks, etc and produce the response based on the intentions.

Here is the Link : 
  https://dzone.com/articles/python-chatbot-project-build-your-first-python-pro

This project contains 6 files : 
<ul/>
  <li>
    Chatbot.h5 -> the Model trained for the Chat bot
  </li>
  <li>
    Chatbot.ipynb -> Model and Data Preprocessing
  </li>
   <li>
    Chatbot.py -> The Graphical User Interface for the Chatbot
  </li>
  <li>
    intents.json -> The dataset for the model
  </li>
   <li>
    tags.pkl -> the dump file of the label of the classification
  </li>
   <li>
    words.pkl -> the cleaned data of each unique word in the dataset
  </li>
<ul/>




### Importing Libraries

```
import nltk
import tensorflow as tf
import json
import numpy as np
import pandas as pd
import pickle
import random
```

### Get the Data from the <i>intents.json</i>

```
file = open('./intents.json')
dataset = json.load(file)
```

```
Output : {"intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day"],
         "responses": ["Hello, thanks for asking", "Good to see you again", "Hi there, how can I help?"],
         "context": [""]
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
         "context": [""]
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"],
         "context": [""]
        }, ...
        ], 
}
```

### Grouping Each of the Patterns, tags, and the Text Vocabularies

```
words = []
documents = []
tags = []

for data in dataset["intents"]:
    for pattern in data['patterns']:
        # Tokenize each word using nltk library
        # Tokenize -> Break the Sentence into words
        word = nltk.word_tokenize(pattern)
        # print(word)
        words.extend(word)
        # Group the Tokenized Word and the Tag
        documents.append((word, data['tag']))
        # Add Tags from the dataset
        if data['tag'] not in tags:
            tags.append(data['tag'])
         
```

### Lemmatize the Word 

from the different word into one word (eg. playing -> Play, Plays -> play)

```
# Lemmatize the data
# Lemmatize -> Grouping the differents words into one word 
lemmatizer = nltk.stem.WordNetLemmatizer()
# Lemmatize the unique word in the words 
words = [lemmatizer.lemmatize(word.lower()) for word in words]
words = sorted(list(set(words)))
```

### Save the labels and the vocabularies in the pickle dump file (.pkl)

```
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))
```

### Set up the training data 
Since the model cannot read the readable text, We need to convert the text into numbers so the model can learn it

```
training = []

for docs in documents:
    bag_of_words = np.zeros(shape=(len(words)), dtype=np.float32)
    tag_pattern = np.zeros(shape=(len(tags)), dtype=np.float32)

    # Get the word in each pattern
    words_pattern = docs[0]
    # Lemmatize the word in each pattern
    words_pattern = [lemmatizer.lemmatize(word.lower()) for word in words_pattern]

    # Get the tag in each pattern
    tag = docs[1]

    # create the bag of words array with 1, if word is found in current pattern
    for word in words_pattern:
        if word in words:
            bag_of_words[words.index(word)] = 1
        else : 
            bag_of_words[words.index(word)] = 0

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    tag_pattern[tags.index(tag)] = 1

    training.append([bag_of_words, tag_pattern])

```

since the word (Hi, There) is on the word vocabulary, the index of the training data is 1 where the index of the same word on the vocabulary (x_train)

```
eg. 

["'s", ',', '?', 'a', 'adverse', 'all', 'anyone', 'are', 'awesome', 'be', 'behavior', 'blood', 'by', 'bye', 'can', 'causing', 'chatting', 'check', 'could', 'data', 'day', 'detail', 'do', 'dont', 'drug', 'entry', 'find', 'for', 'give', 'good', 'goodbye', 'have', 'hello', 'help', 'helpful', 'helping', 'hey', 'hi', 'history', 'hola', 'hospital', 'how', 'i', 'id', 'is', 'later', 'list', 'load', 'locate', 'log', 'looking', 'lookup', 'management', 'me', 'module', 'nearby', 'next', 'nice', 'of', 'offered', 'open', 'patient', 'pharmacy', 'pressure', 'provide', 'reaction', 'related', 'result', 'search', 'searching', 'see', 'show', 'suitable', 'support', 'task', 'thank', 'thanks', 'that', 'there', 'till', 'time', 'to', 'transfer', 'up', 'want', 'what', 'which', 'with', 'you']

['Hi, 'there'] 

[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
0., 0., 0., 0.]

```

and it also applies to the labels (y_train). and since (hi, there) is a greeting so the value of the label (1) is represent the intent in the tags

```
['greeting', 'goodbye', 'thanks', 'options', 'adverse_drug', 'blood_pressure', 'blood_pressure_search', 'pharmacy_search', 'hospital_search']

greeting

[1., 0., 0., 0., 0., 0., 0., 0., 0.]

```

### Create and Train a Model

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(x_train[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64 , activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(y_train[0]) , activation='softmax'),
])

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
    metrics=['accuracy']
)

hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=5, 
    verbose=1,
)

model.save('Chatbot.h5', hist)

```

### Making a Prediction

Clean the Input by Splitting the text and Lemmatize it to a simpler word
```
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
```

Convert the readable text into numbers as the input of the model

```
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
```

Predict the Model using the predict function provided by keras.model (in the prediction it takes the probability of each class as an output)

```
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
    
```

Get the response from the dataset based on the predicted intention of the conversation

```
def get_response(model, sentence):
    res, prob = predict(model, sentence)

    intents_json = intent['intents']
    for intent_json in intents_json:
        if intent_json['tag'] == res:
            response = random.choice(intent_json['responses'])

    return response
```

### Output
![image](https://user-images.githubusercontent.com/43366004/218687539-5a369d5d-c3f4-4654-98fe-2ad781c62355.png)








