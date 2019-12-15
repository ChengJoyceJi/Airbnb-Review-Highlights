from data import Data
from pke.unsupervised import TopicRank
import sys
import os

def main():
    data = Data()
    data.parse_input_data()

    list_id = sys.argv[1]

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
    for (keyphrase, _) in extractor.get_n_best(n=3, stemming=True):
        print(keyphrase)

    os.remove("temp_review_text.txt")

    
    
    
    
if __name__ == '__main__':
    main()