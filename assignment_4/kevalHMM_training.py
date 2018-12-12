'''
CSCI5832: NLP
Professor: James Martin
Assignment 4: Named Entity Recongnition
Student: Keval Shah
Date: December 7 2018
'''

# -*- coding: utf-8 -*-
import os
from collections import Counter
import numpy as np

script_dir = os.path.dirname(__file__)
rel_path = "gene-trainF18.txt"
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

# Split the data into training and test set.
# np.random.shuffle(all_data)
len_all = len(all_data)
len_train = int(len_all*0.8)
len_test = int(len_all-len_train)

training, test = all_data[:len_train], all_data[len_train:]
# training = []
# for unq in not_unique_training:
# 	if unq not in training:
# 		training.append(unq)
# print (len(training),len(not_unique_training))

print ("Total sentences: ", len_all, "Training: ", len_train, "Test: ",len_test)

# Count the occurences of each POS-tag in entire data
tags = Counter()
for sentence in all_data:
	for word in sentence:
		tags[word[2]] += 1
	tags['s'] += 1 # 's' is the tag to indicate start

##	 -------------				BASELINE			--------------


# Maitain the count of each tag assigned to a given word in training set!
uniq_words = {}
for si, sentence in enumerate(training):
	# print (si,"/",len_train)
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
unk_threshold = 3

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
	# print (col_list[s_len-1])
	col_list[s_len-1] = 1
	best_path = col_list

	prediction=[]

	for aaa in best_path:
		prediction.append(tag_words[aaa])
	return (prediction)


'''

			.viterbi Implemented.

			Now apply the tagger and do analysis on your

				predicted genes vs actual genes

			You have to report on the precesion, recall and the f1-score.

'''


# replace the unknown words with 'unk'
goodTest = []
sampleS = len(test)
for sentence in test[0:sampleS]:
	for wii, word in enumerate(sentence):
		if word[1] not in uniq_words.keys():
			sentence[wii][1] = "<unk>"
	goodTest.append(sentence)


# Apply your tagger and then extract the genes that it found
predictedTest = []
for sentence in (goodTest):
	parsedTags = applyViterbi(sentence)
	parsedSentence = []
	for iw, word in enumerate(sentence):
		parsedSentence.append([str(iw+1), word[1], parsedTags[iw]])
	predictedTest.append(parsedSentence)

# extract the genes found using the tagger
foundGenes = {}
for sentence in (predictedTest):
	geneTag = []
	geneName = ''
	for word in sentence:
		if word[2] == 'B':
			# print (sentence)
			geneTag.append(word[2])
			geneName += word[1]
			continue
		if geneTag != []:
			if word[2] == 'I':
				geneTag.append(word[2])
				geneName += "+"+word[1]
			else:
				foundGenes[geneName] = geneTag
				geneTag = []
				geneName = ''

# extract the actual genes from the data
correctGenes = {}
for sentence in (goodTest):
	geneTag = []
	geneName = ''
	for word in sentence:
		if word[2] == 'B':
			# print (sentence)
			geneTag.append(word[2])
			geneName += word[1]
			continue
		if geneTag != []:
			if word[2] == 'I':
				geneTag.append(word[2])
				geneName += "+"+word[1]
			else:
				correctGenes[geneName] = geneTag
				geneTag = []
				geneName = ''

# Find the precision and recall
numerator = 0.0

for aGene in foundGenes.keys():
	if aGene in correctGenes.keys():
		if correctGenes[aGene] == foundGenes[aGene]:
			numerator += 1

recall = numerator/len(correctGenes)
precision = numerator/len(foundGenes)
f1_score = (2*recall*precision)/(recall+precision)

print ("precision:",  precision)
print ("recall:",  recall)
print ("f1_score:",  f1_score)

rel_path = "actualGenesViterbi.txt"
abs_file_path = os.path.join(script_dir, rel_path)
f_out1 = open(abs_file_path,"w")
for si, ansSen in enumerate(goodTest):
	for wi, ansWord in enumerate(ansSen):
		f_out1.write(ansWord[0]+"\t"+ansWord[1]+"\t"+ansWord[2]+"\n")
		# print (ansWord[0]+"\t"+ansWord[1]+"\t"+compareMatrix[si][wi])
	if (si!=len(goodTest)-1):
		f_out1.write("\n")

rel_path = "predictedGenesViterbi.txt"
abs_file_path = os.path.join(script_dir, rel_path)
f_out2 = open(abs_file_path,"w")
for si, ansSen in enumerate(predictedTest):
	for wi, ansWord in enumerate(ansSen):
		f_out2.write(ansWord[0]+"\t"+ansWord[1]+"\t"+ansWord[2]+"\n")
		# print (ansWord[0]+"\t"+ansWord[1]+"\t"+compareMatrix[si][wi])
	if (si!=len(predictedTest)-1):
		f_out2.write("\n")