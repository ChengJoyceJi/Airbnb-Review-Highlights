# Sentiment analysis

import spacy
import nltk
import metapy
import sklearn
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
#import raw data
print("Reading raw data...")
# can be modified
train_data_file = './processedRawData/dataOut_8000.txt'
# read the file into an array, one line per element
training_input = []
for line in open(train_data_file, 'r'):
    training_input.append(line.strip())

# can be modified
train_target_file = './processedRawData/targetOut_8000.txt'
# read the ratings into an array, one line per element
training_target = []
for t in open(train_target_file,'r'):
    training_target.append(int(t))

import dataUtil

X_training, X_validation, y_train, y_val = train_test_split(
    training_input, training_target, train_size = 0.75
)

X_training = dataUtil.basic_preprocess(X_training)
X_training = dataUtil.remove_stop_words(X_training)
X_training = dataUtil.get_lemmatized_text(X_training)

import calendar;
import time;
ts = calendar.timegm(time.gmtime())

print('Fitting ngram...')
from sklearn.feature_extraction.text import CountVectorizer
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))
ngram_vectorizer.fit(X_training)
joblib.dump(ngram_vectorizer,"./trainedModel/ngram_"  + str(len(training_input)) + "_" + str(ts) + ".joblib")
transformed_X_training = dataUtil.ngram_vectorize(X_training,ngram_vectorizer)

print('Training svm...')
svm = LinearSVC(random_state=0,multi_class='ovr')
svm.fit(transformed_X_training, y_train)
# save the trained model into a joblib dump
joblib.dump(svm,"./trainedModel/svm_" + str(len(training_input)) + "_" + str(ts) + ".joblib")

# validation on the trained model
print('Validation...')
transformed_X_validation = dataUtil.one_func_transform(
    X_validation,
    "./trainedModel/ngram_"  + str(len(training_input)) +  "_" + str(ts) + ".joblib",
    )
predictions = svm.predict(transformed_X_validation)


print('Analyzing results...')
differences_map =  [0,0,0,0,0,0]
for i in range(len(predictions)):
    prediction = predictions[i]
    expected = y_val[i]
    difference = abs(expected- prediction)
    if(difference == 5):
        print("\n")
        print("The following case has completely wrong(expect:"
        + str(expected)
        + ", got:"
        + str(prediction)
        + ") prediction:")
        print(X_validation[i])
    differences_map[difference] = differences_map[difference] + 1

for i in range(len(differences_map)):
    print(str(differences_map[i]/len(predictions)*100) + 
    "% predicted ratings differ by "+str(i))

# With 25000 data to train, the models are too big and can't be uploaded to github
# 47.264% predicted ratings differ by 0
# 35.936% predicted ratings differ by 1
# 8.559999999999999% predicted ratings differ by 2
# 3.888% predicted ratings differ by 3
# 2.848% predicted ratings differ by 4
# 1.504% predicted ratings differ by 5