import pickle

from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/emotions', methods=['POST'])
def home():
    model = open("emotions.pkl", 'rb')
    loaded_model = pickle.load(model)
