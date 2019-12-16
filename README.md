# Airbnb-Review-Highlights

## Introduction
We’re presenting a new tool: Airbnb Review Highlights with the purpose of saving people’s time in choosing properties, and discovering some underlying sentiments that can’t simply be inferred from the review scores. The tool includes two features:
1. Generate an average sentiment score of the reviews based on writers’ altitude
2. Generate top three key phrases about different aspects of the listing, e.g. great location, delicious breakfast, room too small, bad service, etc

Target users are
1. P​eople who want save time by getting the most important aspects of the Airbnb place without reading through all the review text
2. People who want to get an unbiasd sentiment socre of the listing

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

## Example
```
(base) Chengs-MacBook-Pro:Airbnb-Review-Highlights jji$ python main.py 36944244
Total # of reviews:  6
Review #1: Totally worth!
Review #2: Great host, exactly what I expected, area is decent and only a 7 minute walk to the train to Manhattan.
Review #3: This place is close to the subway, inexpensive, and good for a place to crash. Unfortunately, there was no a/c while we were there, so it was boiling. If you can't stand intense heat, maybe look somewhere else, or choose to stay here in the cooler months.
Review #4: Eden’s place is very clean. It’s about a one mile walk to the Utica - Crown Heights subway station. Comfy mattress.
Review #5: Host lied about apartment took my money with no refund . when place said available but after contacting her she said not available till 1130 pm scAm do not book scam
Review #6: Quit near with subway station. Really cozy and clean, and a perfect place to stay if you travel with friends
Starting basic preprocess of data
Removing stop words...
Lemmatizing...
Transforming...
==========================================================
average sentiment score:  4.666666666666667
Top three key words (phrases):
perfect place
subway
minut walk
```

## Inplementation
- Remove stop words
- Word stem
- Extract english words only
- Utilized a pre-trained SVM model and ngram library (joblib)
- Use pke library to get the 3 most frequent key phrases


## Contribution
Yufei Zhao - yufeiz7 - data processing, presentation
Cheng Ji - chengj4 - keyword extraction, demo
Hang Zhu - hangzhu2 - sentiment analysis


## References
- Data source: http://insideairbnb.com/get-the-data.html
- Keyphrase extraction: https://github.com/boudinfl/pke
- Sentiment analysis: https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
