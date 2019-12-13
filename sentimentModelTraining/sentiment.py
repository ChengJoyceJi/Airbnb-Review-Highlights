# Sentiment analysis

import spacy
import nltk
import metapy
import sklearn
import joblib


# [STEP 1] IMPORT
print("reading files...")

# train_data_file = './movie_data/full_train.txt'
train_data_file = './allDataOut/dataOut_train.txt'
# test_data_file = './movie_data/full_test.txt'
test_data_file = './allDataOut/dataOut_test.txt'
reviews_train = []
num_of_entries = 25000
i = 0
for line in open(train_data_file, 'r'):
    if(i > num_of_entries):
        break
    reviews_train.append(line.strip())
    i = i +1
    
i = 0
reviews_test = []
for line in open(test_data_file, 'r'):
    if(i > num_of_entries):
        break
    reviews_test.append(line.strip())
    i = i + 1


#[STEP 2] Clean and Preprocess
print('Starting cleaning process...')
import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

train_target_file = './allDataOut/targetOut_train.txt'
test_target_file = './allDataOut/targetOut_test.txt'

train_target = []
i = 0
for t in open(train_target_file,'r'):
    if(i > num_of_entries):
        break
    train_target.append(int(t))
    i = i + 1

test_target = []
i = 0
for t in open(test_target_file,'r'):
    if(i > num_of_entries):
        break
    test_target.append(int(t))
    i = i + 1

# ## [Step 3] Remove stop words
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

reviews_train_clean = remove_stop_words(reviews_train_clean)


## [Step 4] Lemmenization 
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

## [Step 5] n-grams & svm
print('Starting n-gram & SVM...')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))

print('Getting the reviews lemmatized')
lemmatized_reviews_train = get_lemmatized_text(reviews_train_clean)

print('Fiting ngram...')
ngram_vectorizer.fit(lemmatized_reviews_train)

print('Transforming...')
X = ngram_vectorizer.transform(lemmatized_reviews_train)
# X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, train_target, train_size = 0.75
)


# SVM
svm = LinearSVC(random_state=0,multi_class='ovr',C=0.5)
# Naive bayes
# naive_b = GaussianNB()

print('Training...')
# SVM 
svm.fit(X_train, y_train)
# Naive Bayes
# naive_b.fit(X_train, y_train)
joblib.dump(svm,"svm_25000_trigram.joblib")
print('Prediction...')
# SVM
predictions = svm.predict(X_val)
# Naive Bayes
# predictions = naive_b.predict(X_val)


print('Analyzing results...')
differences_map =  [0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(predictions)):
    prediction = predictions[i]
    expected = y_val[i]
    difference = abs(expected- prediction)
    differences_map[difference] = differences_map[difference] + 1

print([d/len(predictions) for d in differences_map])
# print ("Accuracy for C=%s: %s" 
#         % (c, accuracy_score(y_val, svm.predict(X_val))))
    
# Accuracy for C=0.005: 0.8904
# Accuracy for C=0.01: 0.89008
# Accuracy for C=0.05: 0.88976
# Accuracy for C=0.25: 0.88976
# Accuracy for C=0.5: 0.89024

# final_svm_ngram = LinearSVC(C=0.01)
# final_svm_ngram.fit(X, target)
# print ("Final Accuracy: %s" 
#        % accuracy_score(target, final_svm_ngram.predict(X_test)))