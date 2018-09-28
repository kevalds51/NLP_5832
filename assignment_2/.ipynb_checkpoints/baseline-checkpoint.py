'''
NLP
Professor: James Martin
Assignment 2: Part-Of-Speech Tagging
Author: Keval Shah
Date: September 25 2018
'''

# -*- coding: utf-8 -*-
import os
from collections import Counter
import numpy as np

script_dir = os.path.dirname(__file__)
rel_path = "berp-POS-training.txt"
abs_file_path = os.path.join(script_dir, rel_path)

F = open(abs_file_path,"r")

# This will divide the data in list of sentences
# word[0] : word position in sentence
# word[1] : word
# word[2] : POS tag
all_data = []
sentence = []
flag=100
for i in F.readlines():
	if i == '\n':						# if '\n', end sentence
		all_data.append(sentence)
		sentence=[]
	else:
		word = i.split("\t")			# Remove the '\n'
		word[2] = word[2][:-1]
		sentence.append(word)

# Split the data into training and test set.
np.random.shuffle(all_data)
len_all = len(all_data)
len_train = int(len_all*0.8)
len_test = int(len_all*0.1)
len_dev = int(len_all-len_test-len_train)


training, devd, test = all_data[:len_train], all_data[len_train:len_train+len_test], all_data[len_train+len_test:]
print ("Total sentences: ", len_all, "Training: ", len_train, "Development: ", len_dev, "Tested: ",len_test)

# Count the occurences of each POS-tag in entire data
tags = Counter()
for sentence in all_data:
	for word in sentence:
		tags[word[2]] += 1

# Maitain the count of each tag assigned to a given word
uniq_words = {}
for sentence in training:
	for word in sentence:
		if word[1] not in uniq_words.keys():
			uniq_words[word[1]]=Counter()
		uniq_words[word[1]][word[2]]+=1

# test dictionary will store the [predicted value, actual value] for all test words
test_dict = {}
new_word_count = 0
words_tested = 0
incorrect_prediction = 0
for sentence in test:
	for word in sentence:
		words_tested += 1
		if word[1] in uniq_words:
			test_dict[word[1]] = [uniq_words[word[1]].most_common(1)[0][0], word[2]]
			if test_dict[word[1]][0]!=test_dict[word[1]][1]:
				incorrect_prediction += 1
 		else:
			new_word_count += 1

print ("Baseline approach stats")
print ("Words tested: ", words_tested, "New words: ", new_word_count, "Wrongly predicted: ", incorrect_prediction)