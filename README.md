# Twitter Sentiment Analysis Dissertation
 BSc Computer Science Dissertation


Final Year Undergraduate Project 2019/2020
BSc Computer Science
Queen Mary University of London
School of Electronic Engineering and Computer Science





Twitter Sentiment Analysis
PROJECT REPORT








Orlando Renato Antonio Rati
o.r.a.rati@se16.qmul.ac.uk
160567054


Supervised by
Dr. Arman Rezaei Khouzani
arman.khouzani@qmul.ac.uk







11th May 2020





Disclaimer
This report is submitted as a requirement for the degree of BSc Computer Science, part of the School of Electronic Engineering and Computer Science at Queen Mary, University of London. It is the product of my own work and development except where stated otherwise. This report may be copied freely and distributed provided the source is acknowledged.

Acknowledgements
I, the author, would like to thank my supervisor, Dr. Arman Rezaei Khouzani, for his guidance and support during this project which have been an enormous help. I would also like to thank him for his time teaching, specifically the module “Data Mining” that equipped me with fundamental machine learning skills, the motivation behind this idea,  and a foundation of this project. Lastly, I would like to acknowledge Dr. Paulo Oliva for teaching the module “Web Programming” and explaining core concepts on web development which have also helped in developing this project.





Abstract
Sentiment analysis is the use of machine learning techniques, more specifically natural language processing to gauge public sentiment on a certain topic based on text data. This tool is becoming very popular as automation among businesses and financial institutions increases. Millions of users share their opinions daily using social media platforms such as Twitter, Facebook, or Instagram. This makes the platforms a valuable  source of data to perform sentiment analysis on. The outcome of this analysis can then be used to aid investors and businesses, allowing them to tailor their decision-making processes and product development to a specific section of the market based on the results of said analysis. This dissertation will focus particularly on Twitter for the extraction of text data about cryptocurrencies as they are a popular topic of discussion. Twitter is estimated to have around 330 million monthly active users and it restricts its tweets to a limit of 140 characters, which forces its users to mostly express their opinions unambiguously, making it very suitable to extract data from and analyse sentiment.

 
CONTENTS
1. Introduction	6
1.1 Background	7
1.2 Problem Statement	7
1.3 Aim	8
1.4 Objectives	8
2. Literature Review	9
2.1 Sentiment Analysis	9
2.1.1 Sentiment Analysis Applications	9
2.1.2 Sentiment Analysis Challenges	10
2.1.3 Sentiment Analysis Approaches	11
2.2 Twitter	11
2.3 Cryptocurrencies	12
2.4 Web Scraping	13
2.5 Related projects	14
3. Analysis and Design	16
3.1 Requirements Analysis	16
3.1.1 Functional requirements	16
3.1.2 Non-functional requirements	16
3.2 Design	17
4. Implementation	18
4.1 Tools used	18
4.1.1 Jupyter notebook and Anaconda distribution	18
4.1.2 Python	18
4.1.3 Twitter API	18
4.1.4 Multinomial Naive Bayes Classifier	19
4.1.5 Flask Backend	19
4.1.6 Bootstrap Frontend	19
4.1.7 CanvasJS Chart	19
4.1.8 Github for version control and cloud storage	19
4.2 Dataset and data exploration	20
4.3 Model building	22
4.4 Backend of application	24
4.5 Frontend of application	26
5. Results	28
6. Validation	29
7. Evaluation	30
7.1 Requirements evaluation	30
7.2 Model evaluation	30
7.3 Backend evaluation	32
7.4 Frontend evaluation	32
8. Legal, Social, Ethical and Commercial Issues	32
9. Further Improvement	33
9.1 Model improvement	33
9.2 Backend improvement	33
9.3 Frontend improvement	33
9.4 General ideas of improvement	34
10. Conclusion	35
References	36

1. Introduction
This section will provide an introduction to my project including tools and topics of interest.

Sentiment analysis is a natural language processing technique. It aims to analyse text data, which is then classified according to sentiment such as positive/negative or neutral. As opposed to Emotion Analysis, it only analyses the polarity of text and does not identify emotions such as happiness/sadness/anger, etc. 

Cryptocurrencies are digital assets that have become increasingly popular in recent years. As opposed to central banking systems, they use a decentralized control which works through a distributed ledger, typically a blockchain (a list of records linked using cryptography)  that serves as a public financial transaction database (a record stored across multiple entities that acts as a “witness” to transactions hence making cyberattacks more difficult) (Crypto Currency, 2011). Decentralized control means there is no single, central authority that has more power to control the market. This, therefore, attracts people due to its democratic nature as it provides the advantage of funds being secure and not interfered with by a third party such as a bank. The only thing the owner needs to worry about is keeping their private key secure. A private key is a cryptographic concept which in this context allows the owner to access their funds. They are important to all cryptocurrencies as they provide a security mechanism to protect users from thieves and unauthorised accesses to a wallet. 

Twitter is a social media platform for ‘microblogging’ that allows its users to send and receive posts. This works on a follower basis meaning users can follow other users in order for their tweets to appear on their twitter ‘timeline’ (a  feed of “tweets” which includes users and tweet categories they follow). A popular feature is retweeting which allows users to share what other users have tweeted hence allowing for fast circulation and distribution of a post (Ukri.org, 2018). As opposed to other social media platforms such as Facebook or Instagram where users mostly share personal content rather than opinions, Twitter is more popular for sharing news and opinions about those news. Very notable people use Twitter such as Donald Trump whose tweets have caused a lot of controversy.

With those in mind, this project aims to develop a sentiment analysis application that is specific to cryptocurrencies in order to aid decision making of purchasing/selling cryptocurrencies. I have chosen Twitter as my main source due to its large user database and because it reduces bias as opposed to media articles and news. As every user can freely share their opinion publicly rather than a more restricted/controlled source of information such as a newspaper, I feel that this will provide a better approximation of public sentiment. For this project, I will define sentiment to be a “personal feeling” about a topic that can be positive or negative.

1.1 Background
The continuous growth in use and popularity of cryptocurrencies has attracted people into investing in those currencies in hope of significant growth in returns. Bitcoin, for example, the most popular cryptocurrency, has made millionaires. Notable examples include Erick Fineman who borrowed $1000 from his grandmother in 2012 and invested in Bitcoin when the purchase price was $12/Bitcoin. Later, in 2013, Bitcoin reached a price of $1000/Bitcoin granting Erick a return 100 times bigger than what he initially had invested (BIDITEX Exchange, 2019). The continuous volatility of currencies such as Bitcoin and its adoption by global banks such as Goldman Sachs makes it a desirable asset to be investigated especially considering the millionaire stories it comes with. With this in mind, the need for tools to analyse and predict movements are on the rise not just for the individual investor/enthusiast but also for global institutions that are adopting blockchain. (Campbell, 2017)

Fig 1.1 Bitcoin Wallet Address Growth
1.2 Problem Statement
The high volatility of cryptocurrencies has brought along a need for analysing and trying to predict price movements in order to aid investors and investment banks in decision making on purchasing/selling cryptocurrencies. Sentiment analysis is a great tool to obtain public sentiment, therefore, it could aid investment decisions in the same way it aids business decisions to improve products.

1.3 Aim
The primary aim of this project is to serve as a forecasting tool and generate investment advice in combination with other forecasting tools. As a result, it will deliver an application that intends to aid in decision making of whether to purchase/sell cryptocurrency. This will be done by performing sentiment analysis on recently retrieved tweets related to a hashtag that the user will search for. The classification of a tweet will be implemented using machine learning and the trained model will be deployed as a web app that will provide a user interface to allow the user to search for the desired hashtag on which to perform the analysis on.
1.4 Objectives
The main objectives of the project are:
Find a relevant dataset that is already labelled with polarities(sentiment)
Explore the dataset
Preprocess the data to prepare it for training a classifier
Train a classifier
Save the classifier once trained so it can be loaded (not having to train it each time it is ran)
Develop a UI
Deploy the classifier
Allow hashtag searches in the UI which are then classified 
Display sentiment analysis results of the queried hashtag

2. Literature Review
This section will discuss my findings as a result of researching the tools, topics, and related work that are relevant to my project.
2.1 Sentiment Analysis
Sentiment analysis, or opinion mining, is the natural language processing task that aims to extract sentiment from a text in order to understand polarities towards a topic. To identify sentiment, we use natural language processing, statistics, and text analysis to extract, and identify polarities into positive, neutral, and negative. (Agarwal et al., 2011)
2.1.1 Sentiment Analysis Applications
Product and Service Reviews
This is the most common application of sentiment analysis. Companies use this to get a general idea of how their products and services are perceived by the public. There are websites and search engines available which provide automated summaries with the most known being “Google Product Search”.

Result Prediction
Analysing sentiment from good sources can help make predictions on a particular event. An example of this would be elections. Sentiment analysis, in this case, is very valuable to candidates as they can tailor their campaigns and speeches accordingly based on the public’s opinion in order to win voters.
Brand Monitoring
Sentiment analysis is a useful tool for businesses in order to understand customer satisfaction and improve their business strategies. This can be done from reviews, tweets, or other places/discussions where customers mention the brand.
Customer Service
Customer service agents use sentiment analysis to categorize incoming emails into “urgent” or “not urgent”. This aids them in answering urgent queries faster hence providing better customer service which leads to building a better relationship with the customers and a higher quality of service.
Market Research and Analysis
Sentiment analysis is also used in business intelligence to understand potential reasons why consumers are or are not responding to something. Examples would include the reason why someone would or would not buy a certain product.


Stock Market
Sentiment analysis also proves useful in the stock market. This can be especially useful with new product releases. For example, a company may launch a new phone, and based on the public sentiment about their new release, their stock may go up or down.
2.1.2 Sentiment Analysis Challenges
Like emotion recognition, sentiment analysis presents the same challenges. To understand the sentiment of a sentence we first must define what “sentiment” is. Sentiment can be considered categorical and be described as emotions such as happy, angry, sad, etc. However, sentiment can also be described as a dimensional idea where a person does not just feel one way but rather, they are feeling multiple emotions simultaneously.
Communication is a complex topic and verbal devices such as rhetorical questions and sarcasm can mislead sentiment analysis and classify inaccurately. A rhetorical question is one that does not actually need to be answered as the answer is obvious and sarcasm represents the use of irony in a sentence. The only way to really understand the meaning when such devices are used is through context. For example, a person may be talking about a negative topic such as an economic crash and say “How great is this?” or “This is great”. The first example represents a rhetorical question and the second presents sarcasm. Considering the context, an economic crash, it is clear that the sentiment of those texts is negative. However, due to the use of the word “great”, a classifier, if not taking into account context, will incorrectly classify those texts with a positive sentiment, therefore, proving of no use. Although those communication devices may not always be used, they can skew the results of a sentiment analysis problem so it can result in invalid results.
Currently, most sentiment analysis models approach the problem in a categorical way. This is where sentiment is analysed and then classified as belonging to a certain class. The typical classes for a sentiment analysis model are positive, negative and neutral.

2.1.3 Sentiment Analysis Approaches

Fig 2.3 Sentiment Analysis Approaches
The Naïve Bayes classifier is well-known for its applications with natural language processing. Despite it being a simple algorithm, it is able to achieve higher than average performance in tasks such as sentiment analysis. It is also well known that it works well with small data sets which will be the case in this project as tweets are limited to 280 characters. Furthermore, the most common length for a tweet is 33 characters. (Perez, 2018) Therefore, the Naïve Bayes classifier suits the problem and this is the classifier I will be using for this project.
2.2 Twitter
Twitter proves to be a hotspot for social indicator analytics. Twitter currently has 645 million active users and it grows at a rate of 135,000 users per day. However, until 2012, there was no technology available that could scrape and filter Twitter feeds to discern fresh trading data. Social sentiment indicator analysts made good profits once they figured out how to quantify this social media data.
Today, companies such as Dataminr, Datasift, and Social Media Analytics use data analysis technologies to examine Twitter feeds from insiders at publicly traded companies. Bloomberg now adds tweets to its financial data delivery service. They filter tweets from Wall Street analysts, regulators, economists, and government agencies and deliver relevant data to its clients who use this data to stay ahead of the competition when trading stocks.
Twitter’s huge volume of tweets relevant to its investment community has proven profitable for the company. Twitter earned $47.5 million from its data licensing service in 2012 which is a 66% increase compared to its 2011 figure.
Social media analysis relies on algorithms designed with specific features in mind including averages, change, volume, volatility, and dispersion of tweets. These are used by analysts to generate what they call “S-Scores”. These scores are evaluations based on all the above figures that create a sentiment on a given stock over a period of time.
The scores are intended to be used as an indication of whether the social media talk about a certain stock is good or bad news for its price. With this information available, clients can make better-informed decisions when investing in a specific stock based on their sentiment score. (Investopedia, 2019)
2.3 Cryptocurrencies
Cryptocurrencies are digital currencies that use cryptography as their security which makes it nearly impossible to counterfeit or double-spend. Most cryptocurrencies are decentralized and are based on blockchain technology which is a distributed ledger supported by a network of computers. Uniquely, cryptocurrencies are not issued by any central authority which makes them theoretically immune to government interference and manipulation.
Payments are based on virtual “tokens” that are represented by ledger entries internal to the system. Cryptographic techniques are used to safeguard these entries and they include elliptical curve encryption, public-private key pairs as well as hashing functions.
Advantages of cryptocurrencies include ease of use when transferring funds as they remove the need for a third party to be involved such as a bank. These transfers are secured using public and private keys. Users have “wallets” which are account addresses. These have a public key while the private key is only known by the owner and is required when signing transactions. These features allow for fund transfers to be completed with little processing fees meaning users can avoid fees that would usually be in place in the case of financial institutions.
Disadvantages of using cryptocurrencies include transactions in illegal activities. Due to their semi-anonymous nature, they make it easier to engage in activities such as money laundering or tax evasion. (Investopedia, 2019)
Unlike stocks, cryptocurrencies have no exposure to most factors that drive the stock market. Furthermore, they also have no exposure to the returns of currencies and commodities. This can make cryptocurrencies attractive and sentiment analysis could be used to highlight those positive reactions. As a result of this, they would increase in popularity and would be adopted at a faster rate globally.

Cryptocurrency specific price influencing factors include momentum, investor attention, price to dividend, volatility, and supply factors. As an example, Bitcoin is programmed to have a “halving” process take place every 4 years. This process represents a 50% decrease in the production of new bitcoin. This decrease makes new bitcoin more scarce and historically it has been shown that scarcity impacts demand and price. The price of bitcoin increased with both the previous halvings. (Liu and Tsyvinski, 2018)
2.4 Web Scraping
Web scraping refers to a technique used to extract data from the web. In the context of my project, data will refer to text extracted from tweets. While this can manually be done by a user, usually with this term we refer to an automated process implemented by a program as we work with large amounts of data. It is a form of copying and can be done for purposes such as storage or analysis. (Web Scraping Explained, 2020)
Web scraping usage examples include contact scraping which aims to obtain contact details (specifically email addresses) to grow the mailing list of a marketer in order to promote a service/product. Another example includes online price change monitoring or price comparison between different products/services which can be seen on price comparison websites.
In this project, I will be using this technique in order to gather tweets which will then be analysed and classified for sentiment analysis. With Twitter, scraping is made easy as Twitter has developed an API that allows for tweet retrieval with a developer account.

2.5 Related projects
Sentiment analysis is not a novelty but rather a tool that is used very frequently. Furthermore, Twitter is a hotspot of opinions that get posted every second. This allowed me to find similar projects and analyse their approaches in order to get inspired and implement something similar.

One of the projects I researched and most appreciated was “LIVE Sentiment Analysis on Twitter Data using Tweepy, Keras, and Django” by Rohit Agrawal. The project implemented a web application that performs sentiment analysis and classifies tweets as a stream. This means that data is constantly being retrieved as it is posted on twitter, keeping up to date with the most recent tweets. His approach, however, uses Long Short Term Memory, a special kind of Recurrent Neural Networks that have the capability of learning long term dependencies. In the context of text data, this means that words are not just analysed individually but rather they are analysed while taking into account the context in which they are used. This provides an accurate method of analysing sentiment as often context is needed to understand certain feelings. The final accuracy this project achieved for the model’s prediction was 78.4%. I will keep this figure in mind when evaluating my own model for comparison purposes. The backend was implemented using the Django framework and the frontend was implemented using React which uses components to create user interfaces. The advantage React provides is that when a component needs an update (new data coming in), only that component will update hence promoting efficient computation.

The second project that I came across was “Another Twitter sentiment analysis with Python” by Ricky Kim. What I particularly liked about this project was that it actually compared different types of machine learning models, displaying their predictive accuracy and the time required to execute the training and testing time of each. In his findings, the Multinomial Naive Bayes classifier was the fastest to execute while still providing a respectable accuracy figure of 80.09%. With this in mind, I am convinced that approaching this problem with a Naive Bayes classifier will definitely be suitable. Furthermore, this was an interesting study as different approaches were explored. Using statistics, the project managed to create an interesting plot where each word is scored on its positive/negative rate and its frequency. This was done by calculating the harmonic mean of the cumulative distribution function of rates and frequencies. The cumulative distribution function is described by the author as “distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x”. This produced the following plot where every dot represents a word from the vocabulary that was created by extracting words from the twitter dataset:


Fig 2.5 Cumulative Distribution Function Harmonic Mean

Scoring words was an interesting approach and could provide more accurate results and insights into sentiment analysis, expanding the field from the general idea of classifying only based on the 3 classes of positive, negative, and neutral. Having a score would allow a measure of quantification of how positive or how negative the sentiment about something really is.

Those 2 projects had an interesting difference, however. The first one filters out stop words during the data preprocessing while the second does not. I will keep this in mind when building my application and will explore both options to see which will give me better predictive accuracy in my scenario, using a multinomial Naive Bayes classifier.

3. Analysis and Design
This section will talk about the main requirements the application will need to fulfill as well as its design.
3.1 Requirements Analysis
3.1.1 Functional requirements
The software should allow the user to search Twitter for a particular hashtag.
The software should use the Twitter API and retrieve recent tweets.
The software should classify the retrieved tweets into positive/negative.
The software should display a pie chart of the positive/negative sentiment of the queried hashtag.
The software should display a sample of retrieved tweets and what they have been classified as.
3.1.2 Non-functional requirements
The software should be able to run in any browser
The software should change dynamically depending on window size

3.2 Design
The data flow of my application will be as following:
The user interacts with the user interface, inputs a hashtag of choice and submits the query
The hashtag is sent to the backend and it populates the parameter for the Twitter API
The Twitter API search is executed using the parameters
Tweets are retrieved, filtering out retweets and replies
Tweets are processed so that only the message is extracted
An array of messages is populated
Each message from the array is vectorised and classified into positive/negative
A count is kept for each positive and negative messages
A percentage is created for each class(positive or negative)
The results are rendered into a pie chart and displayed on the page
A sample of retrieved tweets is displayed on the page


Figure 3.1 Flow diagram

4. Implementation
This section will describe the implementation of my solution and will be divided into subsections according to different parts of the application.
4.1 Tools used
4.1.1 Jupyter notebook and Anaconda distribution
I have used the Jupyter notebook as my data preprocessing and data exploration environment with Python. This allowed me to integrate code and its output into the same document and it has been helpful for rapid development and visualisation of data.

As good practice, I used Anaconda to create a virtual environment in order to separate this project from others on my system and isolate packages and their versions. This also allows other users to run the app in a virtual environment of their own by just installing the requirements of my project which are saved in a file requirements.txt inside my project folder.
4.1.2 Python
I have used Python as my scripting language as I am fluent in it. Furthermore, I already had experience with machine learning concepts in Python as a result of the data mining module that I have studied. I made use of the following packages in Python:
pandas for data visualisation and manipulation using data frames
BeautifulSoup  for HTML decoding during data preprocessing
re for using regular expressions during data preprocessing
string for accessing string methods during data preprocessing
nltk for tokenization, accessing stopwords and lemmatizing during data preprocessing
numpy for array manipulation during model training
sklearn for feature extraction, splitting data into train/test sets, using a multinomial Naive Bayes classifier and calculating metrics of the classifier
decimal for rounding numbers to 2 decimal places
pickle for saving and loading the classifier

4.1.3 Twitter API
In order to extract tweets, I registered for a developer account with twitter. The standard API allows for retrieval of tweets that can be queried for a particular hashtag. This REST API also provides options such as result_type which can be set to “recent” to retrieve the most recent tweets in the response.

4.1.4 Multinomial Naive Bayes Classifier
The Naive Bayes classifier applies Bayes’ theorem to text, forming classification probabilities. 

Bayes Theorem:

P( c | d )=P( d | c )P(c)P(d)

This calculates the probability of a class (c), given a document (d) which is equal to the (probability of a document (d) given a class (c)  probability of a class (c))  probability of a document (d). We base our classifier on this formula to classify tweets into either positive or negative based on their probability. We assign a tweet to the class which has a higher probability out of the 2.

In the context of my application, it will calculate the probability that a certain text, based on its words as tokens, belongs to a certain class and will classify it accordingly.
4.1.5 Flask Backend
I have used Flask as the framework to implement my backend. Flask is a very light framework in the sense that it has no dependence on external libraries, therefore, it is simple and quick to create an application using it. A functionality of Flask that has greatly helped me in development is the development server it provides with an integrated debugger which allows me to quickly fix bugs and develop.
4.1.6 Bootstrap Frontend
I have used bootstrap as a frontend framework in order to provide access to a library of reusable CSS styles. With Bootstrap, a content delivery network(CDN) can be used to reference the files so they do not have to be stored locally.
4.1.7 CanvasJS Chart
CanvasJS is a HTML5 and Javascript library that provides charts. I have used this library to create a pie chart that is displayed on the results page. Like Bootstrap, this can be referenced using a content delivery network so files do not have to be stored locally.
4.1.8 Github for version control and cloud storage
I have used my personal GitHub account for version control of this project and for cloud storage. This allowed me to revert to previous versions of the project when needed and provided the security of storage as storing my project only locally was risky due to having a single point of failure. Using GitHub however, the project is stored online and can be retrieved at any time should my machine fail.
4.2 Dataset and data exploration
In order to build a model used to predict tweets, I researched available datasets to find labelled data that I used to train my model. 

Initially, I came across a dataset of movie reviews that are included with the NLTK (Natural Language Toolkit) package in Python. However, this dataset only contained 2000 entries (1000 positive and 1000 negative) and it was of movie reviews rather than tweets so I decided to do some more research and find something with more entries and more specific to what I was building.

Further in my research, I came across the dataset Sentiment140 which I downloaded from the Kaggle website. This dataset contains 1,600,000 tweets that are labelled with a polarity column where 0=negative, 2=neutral, and 4=positive. Other columns provided in the dataset were the tweet id, date, flag, and user however those were not relevant to my purpose.

While exploring the dataset, I first wanted to find out how many tweets were in each category of polarity to see if the dataset was fairly split between them:


Figure 4.2 Number of tweets per class

My findings indicated that, in fact, the dataset only contains tweets labelled negative and positive. There were 800,000 entries labelled negative, 800,000 entries labelled positive and 0 entries labelled neutral therefore I proceeded with only 2 classes.

The next step I took in exploring the dataset was to take random samples from the text column in order to decide how to preprocess the text. As guidance, I have used similar projects and observed what techniques they used to clean their data. Those were the projects I researched during my literature review: “LIVE Sentiment Analysis on Twitter Data using Tweepy, Keras, and Django” by Rohit Agrawal and “Another Twitter sentiment analysis with Python” by Ricky Kim.

Figure 4.2.1 Sample texts from tweets

Looking at a sample of 10 entries from the dataset and having looked at the similar projects mentioned, I noticed some features that were not of any value for the purpose of analysing sentiment and therefore I decided to remove them:

Decoding HTML - in a similar project mentioned earlier, “Another Twitter sentiment analysis with Python” by Ricky Kim, samples of data provided text that was still encoded in HTML therefore returned text that contained strings such as “&amp’,’&quot’ and  so on. With this in mind, I decided this should be the first step in cleaning the data as those are  not real words and can therefore provide no predictive power.
“@” mentions - these represent users mentioning other users in their tweets
URL links
Numbers - numbers may, in fact, be important in certain scenarios. For example, 2 sentences where someone would rate something such as “I rate this 10 out of 10” and “I rate this 1 out of 10” would become the same after removing their numbers however one is positive and the latter is negative so this would result in removing important data. In this project, however, I assumed those would be rare instances in tweets and decided to remove them. Furthermore, numbers would act as noise to my approach. Numbers would only be meaningful in models that take into account context such as LSTM (Long Short Term Memory).
Lowercase - in order to identify words as the same, for example, “Good” and “good” they must be exactly the same so I decided to convert the entire dataset to lowercase
Punctuation

Stopwords - I have approached this carefully as it proved to be a controversial topic in my research with very divided opinions. I decided to clean the dataset twice, one containing stopwords and one without and I tested both approaches to see which gave better accuracy to my model (further discussed in the evaluation section).
Lemmatization - I used this to convert words into their base form, therefore, reducing the number of unique words since applying this to “going” for example, would transform it into “go” however this will not be a problem since the word’s meaning does not change

4.3 Model building
After cleaning the dataset and saving it to a new file, it was time to start building my model. However, I noticed that after cleaning the dataset, some text entries were left blank. Those represented tweets that only contained one or more of the features I cleaned such as URL links or “@” mentions for example. I first removed those entries.


Figure 4.3 Count of entries in each column

Once  those entries were removed, it was time to check whether my dataset was still fairly split between tweets labelled as negative and tweets labelled as positive as it was before preprocessing. This was done to ensure that the classifier will not be favouring any class over the other during the training phase therefore removing bias.


Figure 4.3.1 Percentage of negative and positive entries

After achieving a split of almost 50/50 once removing null entries, I determined that the split was sufficiently fair  to continue. Therefore, I started building my model based on this data.


In order to train a model, my textual data had to be converted into numerical form. I have extracted features using the Count Vectorizer provided by the sklearn package. This created a vocabulary of all words and counted their occurrences in each negative and positive tweet.
I have also used the count vectorizer to visualise the most frequent terms that appear in both negative and positive tweets. However, considering I have not removed stop words from the corpus, as expected, the most frequent terms are stop words.


Figure 4.3.2 Displaying the 15 most frequent terms in the dataset and their appearances


During my research, I found that the general opinion is to split the training/test sets in a proportion of 80%/20%. I initially followed this split which gave me 319,248 entries for the test set and the classifier achieved a predictive accuracy of 77.96%.  However, considering the size of my dataset, I decided to split the training and test sets in a proportion of 90%/10%. This gave me 159,624 entries to test my model on which I considered enough. Furthermore, the predictive accuracy of the classifier improved slightly and was now 78%.

Once the model was trained, I saved it to a file using the pickle package. This allows the model to be loaded and used at any time without having to train the model again, therefore, saving time and computational power.
4.4 Backend of application
Using flask’s python decorators, I created routes for my application that map directly to the URLs of the application. My application consists of 2 routes. The default route is “/” which maps to my index.html page and the “/result” route which maps to my results.html page.

In the “/result” route, I define a method called predict(). This method is responsible for the main functionality of the application. First I initialised 2 counter integer variables positive and negative and I set them to 0. Those variables were used to keep a count of positive labelled tweets and negative labelled tweets.

The next step was to create a conditional statement to check that the method of the request is “POST” as that is the method that the submit button triggers when a user searches for a hashtag. If “POST” is the method, I use a string variable named hashtag to collect the input from the text box that was submitted by the user in the form. 

I defined a dictionary named params in which I passed the parameters required for the Twitter API search. Those parameters include the query term for which I used the variable hashtag. Along with the query term I also included 2 options to filter out retweets and replies. I set the count parameter to 100 as this is the maximum, according to the Twitter API documentation. I restricted the language to English and I set the type of result to recent in order to only return the most recent tweets in the response so that the analysis of sentiment will be up to date at the time of the search.

I stored the returned values in a variable named results and retrieved them using the requests.get() method along with the URL provided by the Twitter API, the parameters of the search, and the authentication details of my Twitter developer account. I then converted those to JSON format for easier processing and stored them in a variable named tweets. According to Twitter, their API returns the tweets in the following format:


Figure 4.4 Twitter’s JSON format of tweets

I created an array named messages in which I appended each text from the tweets received. I only extracted the text as this is the only relevant component of a tweet that is required to perform sentiment analysis.

Using a loop to go through the array of messages, at each step, I vectorize the tweet with the count vectorizer I created while building my model and then I use the result to predict it’s label, storing it in the variable prediction. From the dataset, the classes are labelled with a 0 for a negative tweet and a 4 for a positive tweet. Therefore, using a conditional statement, if the prediction returns a 4, I increment the positive counter and I set the label to “Positive”. Contrary, if the prediction returns a 0, I increment the negative counter and set the label to “Negative”. Once this is executed, I update a dictionary that I created named tweet_display and add the tweet along with its label to the dictionary. This allowed me to store the retrieved tweets and their labels as key-value pairs which will be used to display samples of tweets retrieved during the search and what they were labelled as when displaying the results page.

I used a variable total to store the total number of tweets retrieved, a variable posPer to store the percentage of positive tweets retrieved by calculating positive/total*100 and a variable negPer to store the percentage of negative tweets retrieved by calculating negative/total*100. I rounded the percentages to 2 decimal places using the round() method for display purposes.

Lastly, I created a string variable displayMessage to explicitly state the results in a statement form and display it on the page.

The function predict() returns the page results.html and passes it the arguments that were calculated in the function which are hashtag, posPer, negPer, displayMessage and tweet_display.
4.5 Frontend of application
As the main focus of my project is functionality, I decided to create a simple user interface with basic features. I made use of the Bootstrap library and included this in my base.html file, a file that provides a basic HTML template for the other HTML files to inherit. I included bootstrap by referencing their content delivery network (CDN) therefore not having to store files locally.


Figure 4.5 Base HTML page of the application

Making use of the Jinja2 template engine, the base.html file provides a simple template for all other HTML pages to inherit. This is denoted by the notation {% block body%}{% end block %}. This specific example allows inheritance of the scripts referenced in the body in every page as well as the page’s individual content which will have to be wrapped around {% block body %}{% end block %}. This results in a cleaner, easier to read HTML page and avoids code repetition. It is also a way to preserve visual consistency among all pages in the app as a change to base.html would affect all other pages rather than having to edit them individually.

I also included the CanvasJS charting library in my project in order to create the pie chart of sentiment analysis results. This was also included via a content delivery network to avoid storing files locally.


Figure 4.5.1 The JavaScript code used to create and render a pie chart of results

First, I created a container as part of the bootstrap library in which to create my pie chart. I then created a JavaScript function which is triggered when the window loads. Inside this function, I define the variable chart as a CanvasJS.Chart type of object. The parameters of interest are the data points. As my data points, I passed the values of positive and negative which correspond to the values of posPer and negPer from the backend logic. Once the parameters are set, I render the chart and this displays the results obtained from the backend.

5. Results
This section showcases the result of my development.
The development resulted in a simple to use application with a minimalistic user interface, highlighting functionality rather than aesthetics. The application consists of the landing page as index.html with the default route and the results.html with the “/result” route.


Figure 5.1 Index page of the application

The landing page simply consists of a text box in which the user will input the hashtag to search for. Once the hashtag has been inputted and the submit button pressed, this will take the user to the results.html page to display the results of the query.


Figure 5.1.1 Results page of the application


The results.html page displays the CanvasJS pie chart to the user and provides the form that is also present in index.html in order to allow for another search. Furthermore, it displays a sample of tweets and their corresponding prediction that was made by the classifier. 
6. Validation
This section explores testing and potential faults in my application.

During the test phase of my application, I played around with different hashtag inputs. I noticed that querying for an empty string worked and indeed returned a result. This type of search does not provide any value to the application as there would not be a need to search for an empty string ever.

I modified the backend of my application, specifically the app.py file, and implemented the following simple change. Once the hashtag is collected from the text box, I join the string to remove any whitespace present(spaces and tabs) and I use a conditional statement to check if this is a string. If there are no characters present in the string(i.e only whitespace) the conditional will just return the user to the index.html page and display a simple alert "The hashtag cannot be empty!". 


Figure 6.1 Changes made to app.py for validation purposes

At first, I resolved this issue by using the strip() method. However, with further testing, I realised that users were able to search for strings that included spaces for example “bitcoin good”. This does not make sense in the context of my application due to the fact that twitter hashtags cannot contain spaces in them. Therefore, by joining the string and then checking that the string actually contains some characters I was able to resolve both issues. If a user now searches for a string that contains spaces, the string will just be joined into a single word and that will be the hashtag the application searches for. This improves my application as users can now query things that are more specific for example “bitcoin halving” therefore the search is narrowed down and tailored to one’s needs.





7. Evaluation
This section evaluates the project, according to its sections.
7.1 Requirements evaluation
The project was successful and therefore achieved all functional and non-functional requirements stated in the requirements section.
7.2 Model evaluation
To evaluate my model, I have used the metrics of accuracy, precision, and recall. Those metrics are calculated taking into account the following:
True Positives (TP) - the outcomes where the model correctly predicted the positive class
True Negatives (TN) - the outcomes where the model correctly predicted the negative class
False Positives (FP) - the outcomes where the model incorrectly predicted the positive class
False Negatives (FN) - the outcomes where the model incorrectly predicted the negative class

The accuracy represents a measure of the total number of predictions that the model correctly predicts and is calculated as follows:

Accuracy = True Positives + True NegativesTrue Positives + False Positives + True Negatives + False Negatives
Figure 7.2 Calculating accuracy

The precision represents a measure of how precise the model predicts positive labels and is calculated as follows:

Precision = True PositivesTrue Positives + False Positives
Figure 7.2.1 Calculating precision

The recall represents a measure of  actual positives that a model correctly predicted and is calculated as follows:

Recall = True PositivesTrue Positives + False Negatives
Figure 7.2.2 Calculating recall


During the literature review process, the similar projects that I came across had one interesting difference that made me want to investigate myself. One project did not include stop words in the cleaned dataset, while the other did. I also found divided opinions online therefore I preprocessed the original dataset and created 2 new clean datasets. In one dataset I included stop words while in the other I did not. I used both datasets to train a multinomial Naive Bayes classifier with the same training/test split of 90/10 ratio.

The accuracy I achieved when stop words were removed was surprisingly lower than when I did not which is controversial as the general opinion is that stop words do not provide any predictive power and should be removed when performing sentiment analysis. Furthermore, precision and recall were also higher when stop words were not removed.



With stop words
Without stop words
Accuracy
78%
76.85%
Precision
79%
77.38%
Recall
76.22%
76.01%
Figure 7.2.3 Metrics table for classifiers with and without stopwords

As shown in the table, all the metrics I have used scored better when stop words were included and therefore the model used in my application included stop words in its vocabulary.

As a reference, the project “LIVE Sentiment Analysis on Twitter Data using Tweepy, Keras, and Django” by Rohit Agrawal achieved an accuracy of 78.4% using a Long Short Term Memory(LSTM) model. This is only 0.4% higher than the accuracy I achieved using a multinomial Naive Bayes classifier. Considering the LSTM model takes into account context, I expected the figures to have a more significant difference. An LSTM model requires much more computational power and takes significantly longer to train as opposed to the classifier I trained which only required a few minutes. With this in mind and considering that the accuracy figures of both methods have such an insignificant difference, a multinomial Naive Bayes classifier proves to be almost as effective and is definitely more efficient as it comes with the advantage of significantly reduced training time and much simpler implementation.

7.3 Backend evaluation
Using Flask has allowed me to quickly develop the app and deploy my model however I identified a few issues along the way.

Firstly, the retrieved tweets are stored in an array, therefore, my application has no database for storage. While storing the tweets in an array works in this scenario, it prevents further development of the application. On the contrary, if I had used a database this would have allowed me to implement other features to my application. Storing tweets in a database would allow for querying historical data and therefore would increase the functionality of the app. This could serve to show how sentiment of a topic changes over time and could prove useful in a business environment as well to see how the public’s opinion changes with time, as a product is improved for example.
7.4 Frontend evaluation
While very basic and minimalistic, the frontend of my application serves its purpose and provides an easy to use user interface. Visually, it may not necessarily be considered appealing as it consists of very few elements. However, while undertaking this project, my main focus was on the functionality of the application and providing a classifier with good accuracy. 
8. Legal, Social, Ethical and Commercial Issues
This section discusses the issues related to my project.

As this project is concerned with machine learning which is a branch of the artificial intelligence field, the legal aspects in regards to artificial intelligence have to be taken into consideration. The European Commission released “Ethics Guidelines for Trustworthy Artificial Intelligence” on the 8th of April 2019. According to the guidelines, artificial intelligence should respect all applicable laws and regulations, it should respect ethical principles and values and it should be robust and reliable to prevent it from causing any potential harm. The project follows those guidelines and it is lawful, only accessing data that is made available publicly with the consent of its authors which in this case represent Twitter users. The project also respects ethical issues and does not compromise any of them. An example of an ethical issue may be the security of data. However, the project as right now does not store any data about its users hence there is no concern about data security. Lastly, the project is robust in the sense that it cannot cause any harm. As the implementation does not require a physical robot but is rather concerned with data analysis, it is safe and cannot cause any harm therefore it is in accordance with the third guideline. Respecting all guidelines, the project is therefore in accordance with the law.

9. Further Improvement
This section discusses further improvement of ideas that I have for my application.

While I am satisfied with the outcome of my project, I would certainly work to improve it if I had more time. Initially, I struggled to find a labelled tweet dataset and started the project based on a labelled set of movie reviews provided with the NLTK package. Another challenge I faced was the time it required to preprocess the data. Using the Jupyter notebook and Python lambda functions to do this, the process took approximately 5 hours to complete, and due to my research with regards to stop words I had to do this twice which was very time-consuming.
9.1 Model improvement
To further improve classification accuracy, I would research into parameter tuning specifically for the multinomial Naive Bayes classifier. I would also explore different models and statistical methods to assess and score words, similar to the related project I researched, “Another Twitter sentiment analysis with Python” by Ricky Kim where he uses the cumulative distribution function harmonic mean. I would also explore deep learning methods such as the LSTM model that was used in the first project I researched, “LIVE Sentiment Analysis on Twitter Data using Tweepy, Keras, and Django” by Rohit Agrawal. This implementation attracts me the most as it takes into account the context in which a word is used and therefore mimics human capability of identifying sentiment very closely and therefore has the potential for very high accuracy if tuned properly.

Furthermore, I would implement a way that takes into account bigrams and trigrams. Bigrams are phrases consisting of 2 words that require each other in order to make sense such as “San Francisco”. Similarly, trigrams are phrases consisting of 3 words that require each other in order to make sense such as “The Three Musketeers”. This is likely to improve the accuracy of my classifier as it takes into account some context.
9.2 Backend improvement
If I had more time, I would change my project and implement a Django framework backend instead of a Flask one. Django, unlike Flask, which I used, includes an ORM (Object Relational Model) out of the box. This would allow for easy implementation of a database and expansion of my application. With this in mind, if the application’s accuracy would be improved, I could commercialise it and implement user accounts. Those could be used to provide different features and access to different types of users and could work as a service on a subscription basis. With Django, this could be achieved easily as it provides an admin framework and login features out of the box.
9.3 Frontend improvement
As the frontend of my application is very simple, I would further improve it by styling it better to make it more visually appealing and displaying more visual analytics such as how sentiment of a certain topic changed historically over time. A frontend framework I would use is React as this promotes a modular design. It uses components and would be suitable to my application as its functionality allows an individual component to update rather than the whole webpage. This would make the application more efficient. Furthermore, with React, I could implement my application using a single webpage. This could be achieved through the notion of a state which React uses for its components in order to update them.
9.4 General ideas of improvement
Considering the parameters of the Twitter API, the only parameter that I currently allow the user to access is the query. 

The application could be further improved by allowing users to further tailor their search and access other parameters such as result_type where they could choose from 3 options: mixed, recent, or popular. Mixed results return real-time tweets as well as popular ones, popular results return only popular tweets and recent results return only recent tweets. Currently, the functionality of my app only returns recent results as the main goal is to provide up to date results at the time of the search.

Another important parameter is the lang which represents the language that is identified by Twitter in a specific tweet. The language is set to English in my application and this narrows the user audience significantly. To expand the application, I would allow the user to search for other languages or multiple languages at a time.

Those changes could be implemented using checkboxes or drop-down lists in the html that are then passed to the backend and populate the parameter variable that is submitted with the Twitter API call.

Another useful improvement could be to modify the application so that instead of returning recent tweets, it constantly receives a stream of them and updates continuously. This implementation would require the Twitter Streaming API. Furthermore, it would work best with a Django backend as this provides a database out of the box which would be useful to store tweets as they come in and a React frontend as this allows for an independent component update, removing the need to refresh a whole page to display new results as they come in. This is important especially in the context of the application, cryptocurrency, as some traders base their profits on taking advantage of market volatility in the very short term. For this application to be useful to such an audience, it must constantly update itself and it must do so as fast as possible since this style of trading is very time-sensitive.

The last idea I have for improvement would be to include a short description of the hashtag the user searches for on the results page. This could be done by retrieving information from Wikipedia(if it exists for a certain hashtag). This feature was implemented in the project LIVE Sentiment Analysis on Twitter Data using Tweepy, Keras, and Django” by Rohit Agrawal.
10. Conclusion
This section discusses the conclusions drawn from undertaking this project.

To conclude, this dissertation explored the natural language processing technique of sentiment analysis using a multinomial Naive Bayes classifier. The classifier was deployed using a web application to provide a user interface, allowing for ease of use. This allows the user to search for a desired hashtag to perform the analysis on and displays the results as a pie chart, along with retrieved sample tweets and their classification labels from the query. 

With a predictive accuracy of 78%, the classifier I trained could prove useful in a real-world application. As compared with the LSTM model developed by Rohit Agrawal mentioned in previous sections, it can also be concluded that a more complex model is not always better. While the LSTM model did achieve a higher accuracy percentage by a very small figure of 0.4%, the time taken to train such a model and its complexity did not necessarily pay off considering the multinomial Naive Bayes was much simpler and faster to implement and yielded almost the same result in accuracy.

References

Agarwal, A., Xie, B., Vovsha, I., Rambow, O. and Passonneau, R. (2011). Sentiment Analysis of Twitter Data. [online] Association for Computational Linguistics, pp.30–38. Available at: https://www.aclweb.org/anthology/W11-0705.pdf [Accessed 28 Nov. 2019].
Blockchain Works. 2020. Which Major Banks Have Adopted Or Are Adopting The Blockchain? | Blockchain Works. [online] Available at: <https://blockchain.works-hub.com/learn/Which-Major-Banks-Have-Adopted-or-Are-Adopting-the-Blockchain-> [Accessed 10 May 2020].
Forbes.com. 2011. Crypto Currency. [online] Available at: <https://www.forbes.com/forbes/2011/0509/technology-psilocybin-bitcoins-gavin-andresen-crypto-currency.html#5949b09c353e> [Accessed 11 May 2020].
Investopedia. 2020. Can Tweets And Facebook Posts Predict Stock Behavior?. [online] Available at: <https://www.investopedia.com/articles/markets/031814/can-tweets-and-facebook-posts-predict-stock-behavior-and-rt-if-you-think-so.asp> [Accessed 10 May 2020].
Investopedia. 2020. Cryptocurrency. [online] Available at: <https://www.investopedia.com/terms/c/cryptocurrency.asp> [Accessed 10 May 2020].
Kaggle.com. 2020. Sentiment140 Dataset With 1.6 Million Tweets. [online] Available at: <https://www.kaggle.com/kazanova/sentiment140> [Accessed 10 May 2020].
Kumar, P., 2020. An Introduction To N-Grams: What Are They And Why Do We Need Them? - XRDS. [online] XRDS. Available at: <https://blog.xrds.acm.org/2017/10/introduction-n-grams-need/> [Accessed 11 May 2020].
Lexology.com. 2020. Emerging Legal Issues In An AI-Driven World | Lexology. [online] Available at: <https://www.lexology.com/library/detail.aspx?g=4284727f-3bec-43e5-b230-fad2742dd4fb> [Accessed 10 May 2020].
Liu, Y. and Tsyvinski, A. (2018). Risks and Returns of Cryptocurrency. SSRN Electronic Journal.
Medium. 2020. Accuracy, Recall & Precision. [online] Available at: <https://medium.com/@erika.dauria/accuracy-recall-precision-80a5b6cbd28d> [Accessed 10 May 2020].
Medium. 2020. Another Twitter Sentiment Analysis With Python — Part 1. [online] Available at: <https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90> [Accessed 10 May 2020].
Medium. 2020. Another Twitter Sentiment Analysis With Python-Part 2. [online] Available at: <https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-2-333514854913?source=user_profile---------20-----------------------> [Accessed 10 May 2020].
Medium. 2020. Another Twitter Sentiment Analysis With Python — Part 3 (Zipf’S Law, Data Visualisation). [online] Available at: <https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-3-zipfs-law-data-visualisation-fc9eadda71e7?source=user_profile---------19-----------------------> [Accessed 10 May 2020].
Medium. 2020. Another Twitter Sentiment Analysis With Python — Part 4 (Count Vectorizer, Confusion Matrix). [online] Available at: <https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-4-count-vectorizer-b3f4944e51b5?source=user_profile---------17-----------------------> [Accessed 10 May 2020].
Medium. 2020. LIVE Sentiment Analysis On Twitter Data Using Tweepy, Keras, And Django. [online] Available at: <https://towardsdatascience.com/live-sentiment-analysis-on-twitter-data-using-tweepy-keras-and-django-99c344e90488> [Accessed 10 May 2020].
Medium. 2020. Why You Should Avoid Removing STOPWORDS. [online] Available at: <https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52> [Accessed 10 May 2020].
Medium. 2020. 5 People Who Became Rich Because Of Bitcoin, And Their Stories.. [online] Available at: <https://medium.com/datadriveninvestor/5-people-who-became-because-of-bitcoin-and-their-stories-1925ef6934fe> [Accessed 10 May 2020].
Medium. 2020. 12 Graphs That Show Just How Early The Cryptocurrency Market Is. [online] Available at: <https://medium.com/@mccannatron/12-graphs-that-show-just-how-early-the-cryptocurrency-market-is-653a4b8b2720> [Accessed 10 May 2020].
Perez, S. (2018). Twitter’s doubling of character count from 140 to 280 had little impact on length of tweets. [online] TechCrunch. Available at: https://techcrunch.com/2018/10/30/twitters-doubling-of-character-count-from-140-to-280-had-little-impact-on-length-of-tweets/ [Accessed 28 Nov. 2019].
Shaping Europe’s digital future - European Commission. 2020. Ethics Guidelines For Trustworthy AI - Shaping Europe’S Digital Future - European Commission. [online] Available at: <https://ec.europa.eu/digital-single-market/en/news/ethics-guidelines-trustworthy-ai> [Accessed 10 May 2020].
toolkit, I., guidance, S. and it?, W., 2020. What Is Twitter And Why Should You Use It? - Economic And Social Research Council. [online] Esrc.ukri.org. Available at: <https://esrc.ukri.org/research/impact-toolkit/social-media/twitter/what-is-twitter/.> [Accessed 10 May 2020].
Webharvy.com. 2020. Web Scraping Explained. [online] Available at: <https://www.webharvy.com/articles/what-is-web-scraping.html> [Accessed 11 May 2020].
Yousef, A., 2020. Figure 2 Sentiment Classification Techniques.. [online] ResearchGate. Available at: https://www.researchgate.net/figure/Sentiment-classification-techniques_fig1_261875740 [Accessed 16 March 2020].
 





