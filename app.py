import flask
from flask import request, jsonify, render_template

from chat.chatbot import chat_bot

app = flask.Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = chat_bot.predict(text=request.json['text'])
        return jsonify({"data": {"text": text}})
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run()
