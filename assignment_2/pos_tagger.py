'''
NLP
Professor: James Martin
Assignment 2: Part-Of-Speech Tagging
Student: Keval Shah
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
# np.random.shuffle(all_data)
len_all = len(all_data)
len_train = int(len_all*0.8)
len_test = int(len_all-len_train)

training, test = all_data[:len_train], all_data[len_train:]
print ("Total sentences: ", len_all, "Training: ", len_train, "Test: ",len_test)

# Count the occurences of each POS-tag in entire data
tags = Counter()
for sentence in all_data:
	for word in sentence:
		tags[word[2]] += 1
	tags['s'] += 1 # 's' is the tag to indicate start
print (tags)

##	 -------------				BASELINE			--------------


# Maitain the count of each tag assigned to a given word in training set!
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

print ("\n\t\tBaseline approach stats")
print ("\t\t-----------------------\t\t")
print ("\t\tWords tested: ", words_tested, "\n\t\tNew words: ", new_word_count, "\n\t\tWrongly predicted: ", incorrect_prediction)
print ("\t\t-----------------------\t\t\n")


'''
	 -------------				BIGRAM MODEL + VITEBRI			--------------

	Expanding P(W_n/T_n) using Bayes and considering bigrams:-
	Need to put create a probability matrix for (T2/T1)
	Need to put create a probability matrix for (T1/W1)
	Since P(W_n) is constant, we take it out of argMax
'''

# 	enumerate the unique words to maintain the bigram occurence count
word_index = {}		# words are keys
index_words = {}	# indexes are keys
for wi, word in enumerate(uniq_words):
	word_index[word] = wi
	index_words[wi] = word
# 	print (word_index)

# 	enumerate the unique tags to maintain the bigram occurence count
tag_index = {}		# tags are keys
tag_words = {}		# indexes are keys
for wi, word in enumerate(tags):
	tag_index[word] = wi
	tag_words[wi] = word

#	I want the 's' tag to have the last index
s_index = tag_index['s']
last_tag = tag_words[len(tags)-1]
tag_index['s'] = len(tags)-1
tag_index[last_tag] = s_index
tag_words[s_index] = last_tag
tag_words[len(tags)-1] = 's'

# print (tag_words)
# print (tag_index)

'''
	Tag transition probability
	Create the probability matrix for P(tag2/tag1)
	Implemented Add-1 smoothing
'''
nn = len(tags)
t2t1_bgCount = 0
tag2_tag1 = np.zeros(shape=(nn,nn-1)) # don't need 's' in columns
t1_counter = Counter()
for si, sentences in enumerate(training):
	for wi, word in enumerate(sentences):
		if wi == 0:		# adding (tag, 's') for starting point
			t1_counter['s'] += 1
			w2 = tag_index[word[2]]
			w1 = tag_index['s']
			tag2_tag1[w1][w2]+=1
			t2t1_bgCount +=1
		t1_counter[word[2]] += 1
		if wi == len(sentences)-1:
			break
		w1 = tag_index[word[2]]
		w2 = tag_index[sentences[wi+1][2]]
		tag2_tag1[w1][w2]+=1
		t2t1_bgCount +=1

# print ("Total bigram counts: ", t2t1_bgCount)
# if I want the count of (NN after s)
# print (tag2_tag1[tag_index["s"]][tag_index["NN"]])

#	Mapping the count to probability
for xx, row in enumerate(tag2_tag1):
	for yy, ele in enumerate(row):
		# add-1 smoothing
		tag2_tag1[xx][yy] = (ele+1)/(t1_counter[tag_words[xx]]+(nn-1))

'''
	Word likelihood probability
	Create the probability matrix for P(word2/tag2)
	Implemented Add-1 smoothing
'''

nw = len(uniq_words)
w2t2_bgCount = 0
word2_tag2 = np.zeros(shape=(nn-1,nw)) # Don't need 's' here
t2_counter = Counter()
for si, sentences in enumerate(training):
	for wi, word in enumerate(sentences):
		t2_counter[word[2]] += 1
		w2 = word_index[word[1]]
		t2 = tag_index[word[2]]
		word2_tag2[t2][w2]+=1
		w2t2_bgCount +=1

# print ("Total bigram counts: ", w2t2_bgCount)
# if I want the count of ("burrito" for "NN")
# print (word2_tag2[tag_index["NN"]][word_index["burrito"]])

#	Mapping the count to probability
for xx, row in enumerate(word2_tag2):
	for yy, ele in enumerate(row):
		# add-1 smoothing
		word2_tag2[xx][yy] = (ele+1)/(t2_counter[tag_words[xx]]+len(uniq_words))

sum_w2t2 = 0
for row in (word2_tag2):
	for ele in (row):
		sum_w2t2 += ele
print ("Sum of P(W2/T2): ",sum_w2t2)

sum_t2t1 = 0
for row in (tag2_tag1):
	for ele in (row):
		sum_t2t1 += ele
print ("Sum of P(T2/T1): ",sum_t2t1)

'''
		I have the smoothed matrix for: T2/T1 and W2/T2
		Checked the probability matrix by summing all the elements
		TODO: apply vitebri to assign POS 
'''