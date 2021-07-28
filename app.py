
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
# import HashingVectorizer from local dir
from vectorizer import vect
from vectorizer import tokenizer
#from update import update_model

clf = pickle.load(open(os.path.join('pk1_objects','classifier.pkl'),'rb'))

db = os.path.join('reviews.sqlite')

def classify(document):
    label = {0:'negative',1:'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    print(y)
    proba = np.max(clf.predict_proba(X)
    return label[y], proba
def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X,[y])
def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

label, proba = classify('The movie was very very good, I liked it ')
print(label)
print(proba)

app = Flask(__name__)

class ReviewForm(Form):
    moviereview = TextAreaField('',[validators.DataRequired(),validators.length(min=15)])
@app.route('/')
def index2():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form = form)
@app.route('/results',methods = ['POST'])
def results2():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html',content = review, prediction = y, probabilty = round(proba *100 ,2))
    return render_template('reviewform.html',form = form)
@app.route('/thanks', methods = ['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    inv_label = {'negative':0,'positive':1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review,y)
    sqlite_entry(db,review,y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug = True)

#It would make sense to validate the user feedback in the SQLite database prior to the update to make sure that the feedback is valuable information for the classifier.

# see for creating backup code from book

#clf = update_model(db_path = db, model = clf, batch_size = 10000)