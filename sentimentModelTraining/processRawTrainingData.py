# This file contains code to process the raw imdb comment data so that
# they can be used as an input to the sentiment analysis training
import os
root_folder = ["./dataset1","./dataset2"]
# allow the amount of data obtained to be adjustable
# the smaller the number, the faster it is to train
num_of_data = 6000
# evenly obtain data from each folder(because pos folder is all >5, and neg is the opposite)
num_of_data_per_folder = num_of_data // 4
# the data file stores all comments, one per line
data_out = open("./processedRawData/dataOut_" + str(num_of_data_per_folder * 4) + ".txt","w")
# the target file stores the rating of each comment, one per line
target_out = open("./processedRawData/targetOut_" + str(num_of_data_per_folder * 4) + ".txt","w")

def readAndWrite(folder,max_count):
    i = 0
    for child_fname in os.listdir(folder):
        if(i == max_count):
            break;
        i = i + 1
        child_data = open(folder + "/" + child_fname,"r").read()
        # imdb rating of the comment is the number after the '_'
        child_split_name = child_fname.split(".")[0].split("_")
        # imdb rating is 1-10, adjust it to be 0-5
        child_target = int(child_split_name[1]) // 2
        data_out.write(child_data + "\n")
        target_out.write(str(child_target) + "\n")

for dataset in root_folder:
    readAndWrite(dataset + "/pos",num_of_data_per_folder)
    readAndWrite(dataset + "/neg",num_of_data_per_folder)

data_out.close()
target_out.close()