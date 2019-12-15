from data import Data
from pke.unsupervised import TopicRank
import sys
import os
import joblib
import dataUtil
import numpy

def main():
    data = Data()
    data.parse_input_data()

    list_id = sys.argv[1]

    review_list = data.list_id_to_reviews.get(list_id)

    if not review_list:
        print("This listing has no review. Aborting...")
    
    print("Total # of reviews: ", len(review_list))

    # use a pre-trained svm and ngram that used 8000 data to train
    loaded_svm = joblib.load("./trainedModel/svm_8000_1576207965.joblib")
    transformed_input = dataUtil.one_func_transform(
        data.list_id_to_reviews.get(list_id),
        "./trainedModel/ngram_8000_1576207965.joblib"
    )
    predictions = loaded_svm.predict(transformed_input)
    # sentiment score predictions
    print("average sentiment score: ", numpy.mean(predictions))

    with open('temp_review_text.txt', 'w') as txt_file:
        for review in data.list_id_to_reviews.get(list_id):
            txt_file.write(review)

    # create a TopicRank extractor
    extractor = TopicRank()

    # get all review text for the target listing
    extractor.load_document(
        input='temp_review_text.txt',
        language="en"
    )

    # select the keyphrase candidates, for TopicRank the longest sequences of 
    # nouns and adjectives
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})

    # weight the candidates using a random walk. The threshold parameter sets the
    # minimum similarity for clustering, and the method parameter defines the 
    # linkage method
    extractor.candidate_weighting(
        threshold=0.74,
        method='average',
        heuristic='frequent'
    )

    # print the n-highest (10) scored candidates
    print("Top three key words (phrases):")
    for (keyphrase, _) in extractor.get_n_best(n=3, stemming=True):
        print(keyphrase)

    os.remove("temp_review_text.txt") 

    
if __name__ == '__main__':
    main()