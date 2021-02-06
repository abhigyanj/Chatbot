# Imports

import flask
from flask import request, jsonify, render_template

from chat.chatbot import chat_bot

# Setting up flask app
app = flask.Flask(__name__)


# Adding home route
@app.route("/", methods=['GET', 'POST'])
@app.route("/index.html", methods=['GET', 'POST'])
def home():
    """Handles home route"""

    if request.method == 'POST':
        # Predict the entered text on the front-end request
        text = chat_bot.predict(text=request.json['text'])
        return jsonify({"data": {"text": text}})
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run()  # Running app
