from flask import Flask, render_template, request
from flask_cors import CORS
import flask
from newspaper import Article
import urllib
import nltk
import pickle
nltk.download('punkt')


app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__, template_folder='templates')

with open('model.pkl', 'rb') as handle:
     model = pickle.load(handle) 
     

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    
    prediction = model.predict([news])
    return render_template('index.html', prediction_text='Those news are "{}"'.format(prediction[0]))
    


if __name__=="__main__":
    app.run(debug=True,use_reloader=False)
    
    
    
    
