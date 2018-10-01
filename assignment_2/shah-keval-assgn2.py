'''
NLP
Professor: James Martin
Assignment 2: Part-Of-Speech Tagging
Student: Keval Shah
Date: September 27 2018
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
all_data.append(sentence)
# print (len(all_data))

# Split the data into training and test set.
# np.random.shuffle(all_data)
len_all = len(all_data)
len_train = int(len_all)
len_test = int(len_all-len_train)

training, test = all_data[:len_train], all_data[len_train:]
# training = []
# for unq in not_unique_training:
# 	if unq not in training:
# 		training.append(unq)
# print (len(training),len(not_unique_training))

# print ("Total sentences: ", len_all, "Training: ", len_train, "Test: ",len_test)

# Count the occurences of each POS-tag in entire data
tags = Counter()
for sentence in all_data:
	for word in sentence:
		tags[word[2]] += 1
	tags['s'] += 1 # 's' is the tag to indicate start

##	 -------------				BASELINE			--------------


# Maitain the count of each tag assigned to a given word in training set!
uniq_words = {}
for sentence in training:
	for word in sentence:
		if word[1] not in uniq_words.keys():
			uniq_words[word[1]]=Counter()
		uniq_words[word[1]][word[2]]+=1
uniq_words["<unk>"]=Counter()

# print word counts
# for ww in uniq_words.keys():
# 	print (ww, sum(uniq_words[ww].values()))
# print (uniq_words)

'''
	Renaming the words to unk if the counter is < aSmallNumber
'''
unk_threshold = 10

# print (uniq_words)
del_keys = []
for ww in uniq_words.keys():
	if ww == "<unk>":
		continue
	if sum(uniq_words[ww].values())<unk_threshold:
		ccc = uniq_words["<unk>"]
		del_keys.append(ww)
		for icc in uniq_words[ww]:
			uniq_words["<unk>"][icc]+=uniq_words[ww][icc]
for ww in del_keys:
	del uniq_words[ww]

# print word counts after unk threshold
# for ww in uniq_words.keys():
# 	print (ww, sum(uniq_words[ww].values()))
# print (uniq_words)

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

# print ("\n\t\tBaseline approach stats")
# print ("\t\t-----------------------\t\t")
# print ("\t\tWords tested: ", words_tested, "\n\t\tNew words: ", new_word_count, "\n\t\tWrongly predicted: ", incorrect_prediction)
# print ("\t\t-----------------------\t\t\n")


'''
	 -------------				BIGRAM MODEL			--------------

	Expanding P(W_n/T_n) using Bayes and considering bigrams:-
	Need to put create a probability matrix for (T2/T1)
	Need to put create a probability matrix for (T1/W1)
	Since P(W_n) is constant, we take it out of argMax
'''

# 	enumerate the unique words to maintain the bigram occurence count
word_index = {}		# words are keys
index_words = {}	# indexes are keys
for wi, word in enumerate(uniq_words.keys()):
	word_index[word] = wi
	index_words[wi] = word

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
		if word[1] not in uniq_words.keys():
			word[1] = "<unk>"
		t2_counter[word[2]] += 1
		w2 = word_index[word[1]]
		t2 = tag_index[word[2]]
		word2_tag2[t2][w2]+=1
		w2t2_bgCount +=1

#	Mapping the count to probability
for xx, row in enumerate(word2_tag2):
	for yy, ele in enumerate(row):
		# add-1 smoothing
		word2_tag2[xx][yy] = (ele+1)/(t2_counter[tag_words[xx]]+len(uniq_words))

#	Summing the matrix values to check if probability == 1
# sum_w2t2 = 0
# for row in (word2_tag2):
# 	for ele in (row):
# 		sum_w2t2 += ele
# print ("Sum of P(W2/T2): ",sum_w2t2)

# sum_t2t1 = 0
# for row in (tag2_tag1):
# 	for ele in (row):
# 		sum_t2t1 += ele
# print ("Sum of P(T2/T1): ",sum_t2t1)

'''
		I have the smoothed matrix for: T2/T1 and W2/T2
		Checked the probability matrix by summing all the elements
		TODO: apply vitebri to assign POS 

				THE VITEBRI ALGORITHM
'''
def applyViterbi(sentence):
	'''
		Globals:-

			tag       	:	counter for tag occurences
			tag_index	:	gives index of key tag
			tag_words	:	gives tag corresponding to index
			tag2_tag1	:	P(T_i/T_i-1)

			uniq_words	:	counter for unique words and its tags
			word_index	:	gives index of key word
			index_words	:	gives word of key index
			word2_tag2	:	P(W_i/T_i)
	'''
	tag_wos = tags.copy()
	del tag_wos['s']
	tag_wos = tag_wos.keys()

	s_len = len(sentence)
	viterbi = np.zeros((nn-1,s_len)) 	# initialise viterbi table
	v_l = np.zeros((nn-1,s_len))
	backtrack = np.zeros((nn-1,s_len), dtype = int) 		# initialise the best path table
	best_path = np.zeros(s_len); 		# output

	# filling the viterbi table for 1st word
	wi = word_index[sentence[0][1]]
	for ii in tag_wos:
		t2 = tag_index[ii]
		viterbi[t2][0] = tag2_tag1[tag_index['s']][t2] * word2_tag2[t2][wi]

	# filling the viterbi table for rest
	for col in range(1,s_len):
		word = sentence[col]
		wi = word_index[word[1]]
		for row in tag_wos:
			rw = tag_index[row]
			temp_max_val = -1.0
			tem_max_ind = -1.0
			for pv_row in tag_wos:
				prev_row = tag_index[pv_row]	
				vit_ele = tag2_tag1[prev_row][rw] * word2_tag2[rw][wi]
				prod = vit_ele*viterbi[prev_row][col-1]
				if prod > temp_max_val:
					temp_max_val = prod
					tem_max_ind = prev_row
			viterbi[rw][col] = temp_max_val
			backtrack[rw][col] = tem_max_ind

	col_last = s_len-1
	final_col= s_len-1
	col_list=[]
	temp_max_val=-1
	for row in range(nn-1):
		if viterbi[row][col_last] > temp_max_val:
			final_row = row

	while final_col >= 0:
		col_list.append(final_row)
		final_row = backtrack[final_row][final_col]
		final_col = final_col-1

	col_list.reverse()
	col_list[s_len-1] = 8
	best_path = col_list

	prediction=[]
	for aaa in best_path:
		prediction.append(tag_words[aaa])
	return (prediction)

rel_path = "assgn2-test-set.txt"
abs_file_path = os.path.join(script_dir, rel_path)

# This will divide the data in list of sentences
# word[0] : word position in sentence
# word[1] : word
test_all_data = []
test_all_data_copy = []

# read the file to assign tags
Ft = open(abs_file_path,"r")
testSentence = []
for i in Ft.readlines():
	if i == '\n':
		test_all_data.append(testSentence)
		testSentence=[]
	else:
		word = i.split("\t")			# Remove the '\n'
		word[1] = word[1][:-1]
		testSentence.append(word)
test_all_data.append(testSentence)
Ft.close()

# read the file again to print corresponding tags
Ft = open(abs_file_path,"r")
testSentence = []
for i in Ft.readlines():
	if i == '\n':
		test_all_data_copy.append(testSentence)
		testSentence=[]
	else:
		word = i.split("\t")			# Remove the '\n'
		word[1] = word[1][:-1]
		testSentence.append(word)
test_all_data_copy.append(testSentence)
Ft.close()

# np.random.shuffle(test_all_data)
test_len = len(test_all_data)

# replace the unknown words with 'unk'
goodTest = []
for sentence in test_all_data:
	for wii, word in enumerate(sentence):
		if word[1] not in uniq_words.keys():
			sentence[wii][1] = "<unk>"
	goodTest.append(sentence)

# produce predictions
compareMatrix = []
for itr, sentence in enumerate(goodTest):
	compareMatrix.append(applyViterbi(sentence))


rel_path = "shah-keval-assgn2-test-output.txt"
abs_file_path = os.path.join(script_dir, rel_path)
f_out = open(abs_file_path,"w")

# print to file
for si, ansSen in enumerate(test_all_data_copy):
	for wi, ansWord in enumerate(ansSen):
		f_out.write(ansWord[0]+"\t"+ansWord[1]+"\t"+compareMatrix[si][wi]+"\n")
		# print (ansWord[0]+"\t"+ansWord[1]+"\t"+compareMatrix[si][wi])
	if (si!=len(test_all_data_copy)-1):
		f_out.write("\n")
	# print ("")