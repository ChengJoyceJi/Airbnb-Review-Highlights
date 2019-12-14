import csv
from collections import defaultdict

import enchant
d = enchant.Dict("en_US")

class Data(object):
    def __init__(self):
        self.list_id_to_reviews = defaultdict(list)


    def parse_input_data(self):
        with open('reviews.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader) # skip the header
            for (list_id, review_id, date, reviewer_id, _, review) in csv_reader:
                if d.check(review):
                    self.list_id_to_reviews[list_id].append(review)
