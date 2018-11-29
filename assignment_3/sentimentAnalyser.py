'''
CSCI 5832: Natural Language Processing
Fall 18
Professor James Martin

Student name: Keval D. Shah

Assignment 3

In this assignment, you will implement a text classification system for sentiment analysis.
I will provide training data in the form of positive and negative reviews.   
Use this training -data to testelop a system that takes reviews as test cases and
categorizes them as positive or negative. 

'''

# -*- coding: utf-8 -*-
import os
from collections import Counter
import numpy as np
import string
import math

script_dir = os.path.dirname(__file__)
rel_pathN = "hotelNegT-train.txt"
rel_pathP = "hotelPosT-train.txt"
rel_test = "HW3-testset.txt"

abs_file_pathP = os.path.join(script_dir, rel_pathP)
abs_file_pathN = os.path.join(script_dir, rel_pathN)
abs_file_test = os.path.join(script_dir, rel_test)


fp = open(abs_file_pathP, "r")
fn = open(abs_file_pathN, "r")
ft = open(abs_file_test, "r")

positives = []
postivesD = {}

negatives = []
negativesD = {}

tests = []
testsD = {}

for line in fp.readlines():
	positives.append(line)
	postivesD[line[0:7]] = line[8:-2]

for line in fn.readlines():
	negatives.append(line)
	negativesD[line[0:7]] = line[8:-2]

for line in ft.readlines():
	tests.append(line)
	testsD[line[0:7]] = line[8:-2]

'''

negatives: List of negative training sentences
positives: List of positive training sentences

tests: List of test sentences

+D: Dicitionary of corresponding sentences

/n and /r were removed -while parsing below

'''

train_allPosWords = []
train_allNegWords = []
test_allWords = []

# I want to retain the '/' punctuation
punctutations = string.punctuation[:14] + string.punctuation[15:]

for review in postivesD.values():
	review = review.translate(None, punctutations)
	train_allPosWords += review.split(" ")

train_posWordSet = set(train_allPosWords)
train_posWords = dict.fromkeys(train_posWordSet, 0)

for word in train_allPosWords:
	train_posWords[word] += 1

for review in negativesD.values():
	review = review.translate(None, punctutations)
	train_allNegWords += review.split(" ")

train_negWordSet = set(train_allNegWords)
train_negWords = dict.fromkeys(train_negWordSet, 0)

for word in train_allNegWords:
	train_negWords[word] += 1

for review in testsD.values():
	review = review.translate(None, punctutations)
	test_allWords += review.split(" ")

test_WordSet = set(test_allWords)
test_Words = dict.fromkeys(test_WordSet, 0)

for word in test_allWords:
	test_Words[word] += 1

'''

train_allPosWords: list of all words
train_allNegWords: : list of all words
train_posWordSet: Set of all unique words
train_negWordSet: Set of all unique words
train_posWords: final Dicitionary of words
train_negWords: final Dicitionary of words

test_allWords: list of all words
test_WordSet: set of all unique words
test_Words: final Dicitionary of words

'''


'''
Given a new review I want to classify it as positive or negative with some confidence,
that is with some probability based on the words. I want to multiply the probability of 
of -each word belonging a given class and get the final probability. I will make the use of
Naive Bayes for this.

'''

countPosWords = float(sum(train_posWords.values()))
countNegWords = float(sum(train_negWords.values()))

# Are lists needed? YES, we need to access words for Bayes!
outputList = []
for review in tests:
	ID = review[0:7]

	review = review[8:-2].translate(None, punctutations).split(" ")

	pPos = 0
	pNeg = 0

	for word in review:
		# The model for positive class
		if word in train_posWords.keys():
			pPos += math.log((train_posWords[word]+1)/(countPosWords+len(train_posWordSet)))
		else:
			pPos += math.log(1/(countPosWords+len(train_posWordSet)))

		# The model for positive class
		if word in train_negWords.keys():
			pNeg += math.log((train_negWords[word]+1)/(countNegWords+len(train_negWordSet)))
		else:
			pNeg += math.log(1/(countNegWords+len(train_negWordSet)))
	# print (pPos,pNeg)
	if pPos > pNeg:
		outputList.append(ID + "\t" + "POS")
	else:
		outputList.append(ID + "\t" + "NEG")

# print ("Size of test-set("+ str(len(tests)) + ") with " + "train-set size (" + str(len(positives)+len(negatives)) + ")")

rel_path = "shah-keval-assgn3-out.txt"
abs_file_path = os.path.join(script_dir, rel_path)
f_out = open(abs_file_path,"w")

# print to file
for si, ansSen in enumerate(outputList):
	f_out.write(ansSen)
	if (si!=len(outputList)-1):
		f_out.write("\n")
f_out.write("\n")