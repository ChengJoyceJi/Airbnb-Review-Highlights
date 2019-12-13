example_input = [
    "I definitely enjoyed my stay at ROW NYC! The location is basically right around the corner from Times Square so you’re super close to all the action. The Kitchen (which is connected to ROW NYC) has amazing food choices, & if you need a drink to get loose District M is literally in the lobby of the hotel. There are several broadway theaters, bars, clothing stores, food spots, and everything else nearby it’s impossible not to have a good time staying at ROW NYC.",
    "DO NOT BOOK THROUGH AirBnb. Do it through the hotel and it will be cheaper that way. The extra fees and taxes they charge you are not worth it.",
    "Great location for a great price. But didn’t feel that clean, can tell a lot of people have stayed here. I would stay here again if it was the only option.",
    "Not that clean but awesome location! The location was the best thing I would definitely book again!",
    "Would not stay here even for the low rate",
    "I really hated this place",
    "This place was wonderful",
    "Any place is better than this",
    "What a joke",
    "The location is good, but the chicken was raw",
    "Look at the toilet, it's a better place"
]

import joblib
import dataUtil
# this example uses a pre-trained svm and ngram that used 8000 data to train
loaded_svm = joblib.load("./trainedModel/svm_8000_1576207965.joblib")
transformed_input = dataUtil.one_func_transform(example_input,"./trainedModel/ngram_8000_1576207965.joblib")
predictions = loaded_svm.predict(transformed_input)
print(predictions)