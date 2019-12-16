# Airbnb-Review-Highlights

## Installation
Please use python 3.7

Install dependcies:
```
pip install git+https://github.com/boudinfl/pke.git
python -m nltk.downloader stopwords
python -m nltk.downloader universal_tagset
python -m spacy download en # download the english model
pip install -U scikit-learn scipy matplotlib
```
Then you can run the code to get sentiment scores key phrases from the review text:
```
python main.py <listing_id>
```
For example
```
> python main.py 38241551
Total # of reviews:  2
Starting basic preprocess of data
Removing stop words...
Lemmatizing...
Transforming...
average sentiment score:  4.5
Top three key words (phrases):
great location
place
nice furnish
```

## References
- Data source: http://insideairbnb.com/get-the-data.html
- Keyphrase extraction: https://github.com/boudinfl/pke
- Sentiment analysis: https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
