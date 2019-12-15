# Airbnb-Review-Highlights

## Installation
Please use python 3.7

To pip install pke from github:
```
pip install git+https://github.com/boudinfl/pke.git
```
pke also requires external resources that can be obtained using:
```
python -m nltk.downloader stopwords
python -m nltk.downloader universal_tagset
python -m spacy download en # download the english model
```
Then you can run the code to get key phrases from the review text:
```
python main.py <listing_id>
```
For example
```
> python main.py 38241551
great locat
place
nice furnish
```


## References
Data source: http://insideairbnb.com/get-the-data.html
Keyphrase extraction: https://github.com/boudinfl/pke