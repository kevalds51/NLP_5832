'''

CSCI 5832: Natural Language Processing
Fall 18
Professor James Martin

Student name: Keval D. Shah

Assignment 3

In this assignment, you will implement a text classification system for sentiment analysis.
I will provide training data in the form of positive and negative reviews.   
Use this training -data to develop a system that takes reviews as test cases and
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

abs_file_pathP = os.path.join(script_dir, rel_pathP)
abs_file_pathN = os.path.join(script_dir, rel_pathN)

fp = open(abs_file_pathP, "r")
fn = open(abs_file_pathN, "r")

positives = []
postivesD = {}

negatives = []
negativesD = {}

testPositives = []
testPostivesD = {}

testNegatives = []
testNegativesD = {}

thisIter = 0
for line in fp.readlines():
	thisIter += 1
	if thisIter % 5 == 0:
		testPositives.append(line)
		testPostivesD[line[0:7]] = line[8:-2]
		continue
	positives.append(line)
	postivesD[line[0:7]] = line[8:-2]

thisIter = 0
for line in fn.readlines():
	thisIter += 1
	if thisIter % 5 == 0:
		testNegatives.append(line)
		testNegativesD[line[0:7]] = line[8:-2]
		continue
	negatives.append(line)
	negativesD[line[0:7]] = line[8:-2]

# print ((positives), len(testPositives), len(negatives), len(testNegatives))

'''

negatives: List of negative training sentences
positives: List of positive training sentences

testPositives: List of positive training sentences
testNegatives: List of negative training sentences

+D: Dicitionary of corresponding sentences

How do I deal with /x?

/n and /r were removed -while parsing below

'''

train_allPosWords = []
train_allNegWords = []
test_allPosWords = []
test_allNegWords = []

# I want to retain the '/' punctuation
punctutations = string.punctuation[:14] + string.punctuation[15:]

for review in postivesD.values():
	review = review.translate(None, punctutations)
	train_allPosWords += review.split(" ")

train_posWordSet = set(train_allPosWords)
train_posWords = dict.fromkeys(train_posWordSet, 0)

for word in train_allPosWords:
	train_posWords[word] += 1
# print (train_posWords)


for review in negativesD.values():
	review = review.translate(None, punctutations)
	train_allNegWords += review.split(" ")

train_negWordSet = set(train_allNegWords)
train_negWords = dict.fromkeys(train_negWordSet, 0)

for word in train_allNegWords:
	train_negWords[word] += 1
# print (train_negWords)



for review in testPostivesD.values():
	review = review.translate(None, punctutations)
	test_allPosWords += review.split(" ")

test_posWordSet = set(test_allPosWords)
test_posWords = dict.fromkeys(test_posWordSet, 0)

for word in test_allPosWords:
	test_posWords[word] += 1

# print (test_posWords)

for review in testNegativesD.values():
	review = review.translate(None, punctutations)
	test_allNegWords += review.split(" ")

test_negWordSet = set(test_allNegWords)
test_negWords = dict.fromkeys(test_negWordSet, 0)

for word in test_allNegWords:
	test_negWords[word] += 1

# print (test_negWords)

'''

train_allPosWords: list of all words
train_allNegWords: : list of all words
train_posWordSet: Set of all unique words
train_negWordSet: Set of all unique words
train_posWords: final Dicitionary of words
train_negWords: final Dicitionary of words

test_allPosWords: list of all words
test_allNegWords: list of all words
test_posWordSet: set of all unique words
test_negWordSet: set of all unique words
test_posWords: final Dicitionary of words
test_negWords: final Dicitionary of words


'''

'''
Given a new review I want to classify it as positive or negative with some confidence,
that is with some probability based on the words. I want to multiply the probability of 
of -each word belonging a given class and get the final probability. I will make the use of
Naive Bayes for this.

'''

# test1 = testPostivesD
tests = testPositives + testNegatives

countPosWords = float(sum(train_posWords.values()))
countNegWords = float(sum(train_negWords.values()))

# Are lists needed? YES, we need to access words for Bayes!
for review in tests:
	ID = review[0:7]
	label = "NEG"
	if ID in testPostivesD.keys():
		label = "POS"

	review = review[8:-2].translate(None, punctutations).split(" ")

	pPos = 0
	pNeg = 0

	for word in review:
		# print (word in train_posWords.values(), word in train_negWords.values())

		# The model for positive class
		if word in train_posWords.keys():
			pPos += math.log(train_posWords[word]/countPosWords)
		else:
			pPos += math.log(1/countPosWords)

		# The model for positive class
		if word in train_negWords.keys():
			pNeg += math.log(train_negWords[word]/countNegWords)
		else:
			pNeg += math.log(1/countNegWords)

	print (pPos,pNeg)
	if pPos > pNeg:
		print (ID + " with label: " + label + " was predicted as: " + "POS")
	else:
		print (ID + " with label: " + label + " was predicted as: " + "NEG")