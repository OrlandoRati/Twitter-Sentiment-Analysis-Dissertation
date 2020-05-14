from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from requests_oauthlib import OAuth1
import requests
from bs4 import BeautifulSoup
from decimal import Decimal

#Load previously trained classifier and count vectorizer
classifier=pickle.load(open("multinomialNBSW.pickle","rb"))
cvec=pickle.load(open("cvec.pickle","rb"))

app = Flask(__name__)

# Parameters needed to access authenticate with the Twitter API using my Twitter developer account
auth_params = {
    'app_key':'a795FUrBy1W0vqJGN9OOEyuOe',
    'app_secret':'iCDdh6t5SzcgXDxwAKALqhTJU2O8Iqdd5W1WHRwuI2NzzeJKsX',
    'oauth_token':'2859033995-JJuwtRge2GcZwOdE8ixqwGCB44MBRJvaG0f9H5j',
    'oauth_token_secret':'zNGcFKS0KTwQUD7oRW62sqJ6lvRajUFwRNi3GD7Dyl2sP'
}

# Creating an OAuth Client connection to connect to the Twitter API
auth = OAuth1 (
    auth_params['app_key'],
    auth_params['app_secret'],
    auth_params['oauth_token'],
    auth_params['oauth_token_secret']
)

# URL according to Twitter API to retrieve tweets
url_rest = "https://api.twitter.com/1.1/search/tweets.json"

# Default route
@app.route("/") # default route
def index():
	return render_template("index.html", result="" )

@app.route("/result", methods = ["POST","GET"])
def predict():
    positive=0
    negative=0

    if request.method == "POST":
        hashtag=request.form["text"]
        hashtag=''.join(hashtag.split())
        if not hashtag:
            alert="The hashtag cannot be empty!"
            return render_template('index.html', alert=alert)
        # Parameters for the query of the tweets
        params = {'q': hashtag + str(" -filter:retweets -filter:replies"), 'count': 100, 'lang': 'en',  'result_type': 'recent'}
        results = requests.get(url_rest, params=params, auth=auth)

        # Converting the results to JSON for easier processing
        tweets = results.json()

        #Creating an array of tweets
        messages = [BeautifulSoup(tweet['text'], 'html.parser').get_text() for tweet in tweets['statuses']]

        tweet_display={}

        for message in messages:
            cvec_result=cvec.transform([message])
            prediction=classifier.predict(cvec_result)
            if prediction == 4:
                positive+=1
                label="Positive"
            else:
                negative+=1
                label="Negative"
            tweet_display.update({message : label})

        total=positive+negative
        posPer=round(positive/total*100,2)
        negPer=round(negative/total*100,2)
        displayMessage="There were " + str(posPer) + " % positive tweets, " + str(negPer) + " % negative tweets with the hashtag " + hashtag

        return render_template("results.html", hashtag=hashtag, positive=posPer, negative=negPer, messagedisplay=displayMessage, display=tweet_display )
    else:
        return render_template("index.html")

if __name__ == "__main__":
	app.debug = True
	app.run()
