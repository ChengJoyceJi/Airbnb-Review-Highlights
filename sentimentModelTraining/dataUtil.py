def basic_preprocess(data):
    import re
    print("Starting basic preprocess of data")
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    data = [REPLACE_NO_SPACE.sub("", line.lower()) for line in data]
    data = [REPLACE_WITH_SPACE.sub(" ", line) for line in data]
    return data

def remove_stop_words(data):
    print('Removing stop words...')
    from nltk.corpus import stopwords
    english_stop_words = stopwords.words('english')
    removed_stop_words = []
    for d in data:
        removed_stop_words.append(
            ' '.join([word for word in d.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

def get_lemmatized_text(data):
    print('Lemmatizing...')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in data]

def ngram_vectorize(data,vectorizer):
    print('Transforming...')
    return vectorizer.transform(data)

# this function transforms a piece of data to 
# a form that is acceptable by the sentiment model
def one_func_transform(data,ngram_path):
    import joblib
    data = basic_preprocess(data)
    data = remove_stop_words(data)
    data = get_lemmatized_text(data)
    ngram_vectorizer = joblib.load(ngram_path)
    return ngram_vectorize(data,ngram_vectorizer)
