import os
from flask import Flask, render_template, request
from sklearn.externals import joblib


app = Flask(__name__)
sgd_clf = joblib.load('./application/models/sgd_clf.pkl') 


@app.route('/')
def index():
	return render_template('index.html'), 200


@app.route('/classify', methods=['POST'])
def classify():
	text = request.form['text']
	
	try:
		assert len(text) > 0
	except:
		return render_template('400.html'), 400

	sentiment = 'positive' if sgd_clf.predict([text]) == ['pos'] else 'negative'

	return render_template(
		'classify_result.html', 
		text=text,
		sentiment=sentiment), 200