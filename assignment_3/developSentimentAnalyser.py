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

devPositives = []
devPostivesD = {}

devNegatives = []
devNegativesD = {}

thisIter = 0
for line in fp.readlines():
	thisIter += 1
	if thisIter % 5 == 0:
		devPositives.append(line)
		devPostivesD[line[0:7]] = line[8:-2]
		continue
	positives.append(line)
	postivesD[line[0:7]] = line[8:-2]

thisIter = 0
for line in fn.readlines():
	thisIter += 1
	if thisIter % 5 == 0:
		devNegatives.append(line)
		devNegativesD[line[0:7]] = line[8:-2]
		continue
	negatives.append(line)
	negativesD[line[0:7]] = line[8:-2]

# print ((positives), len(devPositives), len(negatives), len(devNegatives))

'''

negatives: List of negative training sentences
positives: List of positive training sentences

devPositives: List of positive training sentences
devNegatives: List of negative training sentences

+D: Dicitionary of corresponding sentences

How do I deal with /x?

/n and /r were removed -while parsing below

'''

train_allPosWords = []
train_allNegWords = []
dev_allPosWords = []
dev_allNegWords = []

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



for review in devPostivesD.values():
	review = review.translate(None, punctutations)
	dev_allPosWords += review.split(" ")

dev_posWordSet = set(dev_allPosWords)
dev_posWords = dict.fromkeys(dev_posWordSet, 0)

for word in dev_allPosWords:
	dev_posWords[word] += 1

# print (dev_posWords)

for review in devNegativesD.values():
	review = review.translate(None, punctutations)
	dev_allNegWords += review.split(" ")

dev_negWordSet = set(dev_allNegWords)
dev_negWords = dict.fromkeys(dev_negWordSet, 0)

for word in dev_allNegWords:
	dev_negWords[word] += 1

# print (dev_negWords)

'''

train_allPosWords: list of all words
train_allNegWords: : list of all words
train_posWordSet: Set of all unique words
train_negWordSet: Set of all unique words
train_posWords: final Dicitionary of words
train_negWords: final Dicitionary of words

dev_allPosWords: list of all words
dev_allNegWords: list of all words
dev_posWordSet: set of all unique words
dev_negWordSet: set of all unique words
dev_posWords: final Dicitionary of words
dev_negWords: final Dicitionary of words


'''

'''
Given a new review I want to classify it as positive or negative with some confidence,
that is with some probability based on the words. I want to multiply the probability of 
of -each word belonging a given class and get the final probability. I will make the use of
Naive Bayes for this.

'''

# dev1 = devPostivesD
devs = devPositives + devNegatives

countPosWords = float(sum(train_posWords.values()))
countNegWords = float(sum(train_negWords.values()))

# Are lists needed? YES, we need to access words for Bayes!
correct = 0.0
for review in devs:
	ID = review[0:7]
	label = "NEG"
	if ID in devPostivesD.keys():
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
	# print (pPos,pNeg)

	if pPos > pNeg:
		print (ID + " with label: " + label + " was predicted as: " + "POS")
		if label == "POS":
			correct += 1
	else:
		print (ID + " with label: " + label + " was predicted as: " + "NEG")
		if label == "NEG":
			correct += 1

print ("Accuracy on dev-set("+ str(len(devs)) + ") with " + "train-set(" + str(len(positives)+len(negatives)) + ") : " + str(round(100*correct/len(devs), 2)) + " %")
'''
print("train_allPosWords: ", len(train_allPosWords))
print("train_allNegWords: ", len(train_allNegWords))
print("train_posWordSet: ", len(train_posWordSet))
print("train_negWordSet: ", len(train_negWordSet))
print("train_posWords: ", len(train_posWords))
print("train_negWords: ", len(train_negWords))
print("dev_allPosWords: ", len(dev_allPosWords))
print("dev_allNegWords: ", len(dev_allNegWords))
print("dev_posWordSet: ", len(dev_posWordSet))
print("dev_negWordSet: ", len(dev_negWordSet))
print("dev_posWords: ", len(dev_posWords))
print("dev_negWords: ", len(dev_negWords))
'''