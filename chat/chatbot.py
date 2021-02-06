import os
import json
import pickle
import random

import nltk
import numpy
import tflearn
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops


class ChatBot:
    def __init__(self):
        self.stemmer = LancasterStemmer()
        self.data = None
        self.model = None
        self.words = None
        self.labels = None
        self.output = None
        self.training = None
        self.model_saved = False
        self.chat_history = []
        self.loaded_intent = False
        self.model_trained = False
        self.loaded_model_data = False

    def load_data(self, file_name="data.pickle"):
        with open(file_name, "rb") as data:
            self.words, self.labels, self.training, self.output = pickle.load(
                data)

        self.loaded_model_data = True

        ops.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(
            net, len(self.output[0]), activation="softmax")
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)

    def load_intents(self, file_name="intents.json"):
        with open(file_name, "r") as file:
            self.data = json.load(file)

    def make_data(self):
        self.words = []
        self.labels = []
        docs_x = []
        docs_y = []

        for intent in self.data["intents"]:
            for pattern in intent["patterns"]:
                temp_words = nltk.word_tokenize(pattern)
                self.words.extend(temp_words)
                docs_x.append(temp_words)
                docs_y.append(intent["tag"])

            if intent["tag"] not in self.labels:
                self.labels.append(intent["tag"])

        self.words = [self.stemmer.stem(w.lower())
                      for w in self.words if w != "?"]
        self.words = sorted(list(set(self.words)))

        self.labels = sorted(self.labels)

        self.training = []
        self.output = []

        out_empty = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            temp_words = [self.stemmer.stem(w.lower()) for w in doc]

            for w in self.words:
                if w in temp_words:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[self.labels.index(docs_y[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)

        self.training = numpy.array(self.training)
        self.output = numpy.array(self.output)

        self.loaded_intent = True
        self.loaded_model_data = True

        ops.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(
            net, len(self.output[0]), activation="softmax")
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)

    def save_data(self, file_name="data.pickle"):
        with open(file_name, "wb") as data:
            pickle.dump((self.words, self.labels,
                         self.training, self.output), data)

    def make_model(self, epochs, batch_size=8, show_metric=False):
        self.model.fit(self.training, self.output, n_epoch=epochs,
                       batch_size=batch_size, show_metric=show_metric)

        self.model_trained = True

    def save_model(self, file_name="model.tflearn"):
        self.model.save(file_name)
        self.model_saved = True

    def show_metrics(self):
        print("---------")
        print("METRICS:")
        print(f"Model Saved: {self.model_saved}")
        print(f"Loaded Intents File: {self.loaded_intent}")
        print(f"Model Trained: {self.model_trained}")
        print(f"Loaded Model Data: {self.loaded_model_data}")
        print("---------")

    def _bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return numpy.array(bag)

    def load_model(self, file_name="model.tflearn"):
        self.model.load(file_name)

    def predict(self, text):
        results = self.model.predict(
            [self._bag_of_words(text, self.words)])[0]
        results_index = numpy.argmax(results)
        tag = self.labels[results_index]

        if results[results_index] > 0.5:
            for tg in self.data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

                    resp = random.choice(responses)
                    return resp

        else:
            return "I didn't understand"

    def chat(self):
        print("Start talking with the bot (type quit to stop)!")
        self.chat_history.append(
            "Chatbot -" + "Start talking with the bot (type quit to stop)!")

        while True:
            inp = input("You: ")
            self.chat_history.append("User -" + inp)
            if inp.lower() == "quit":
                break

            results = self.model.predict(
                [self._bag_of_words(inp, self.words)])[0]
            results_index = numpy.argmax(results)
            tag = self.labels[results_index]

            if results[results_index] > 0.4:
                for tg in self.data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                resp = random.choice(responses)
                print(resp)
                self.chat_history.append("Chatbot -" + resp)

            else:
                print("I didn't understand")
                self.chat_history.append("Chatbot -" + "I didn't understand")

    def save_chat(self, file_name):
        with open(file_name, "a") as chat_file:
            chat_file.write("\n".join(self.chat_history))


chat_bot = ChatBot()

chat_bot.load_intents(
    file_name=os.path.join("chat", "intents.json"))
chat_bot.load_data(file_name=os.path.join("chat", "data.pickle"))
chat_bot.load_model(file_name=os.path.join("chat", "model.tflearn"))


if __name__ == '__main__':
    chat_bot.chat()
