'''

CSCI 5832: Natural Language Processing
Fall 18
Professor James Martin

Student name: Keval D. Shah

Assignment 3

In this assignment, you will implement a text classification system for sentiment analysis.
I will provide training data in the form of positive and negative reviews.   
Use this training data to develop a system that takes reviews as test cases and
categorizes them as positive or negative. 

'''

# -*- coding: utf-8 -*-
import os
from collections import Counter
import numpy as np
import string

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

for line in fp.readlines():
	positives.append(line)
	postivesD[line[0:7]] = line[8:-2]

for line in fn.readlines():
	negatives.append(line)
	negativesD[line[0:7]] = line[8:-2]




'''

How do I deal with /x?

/n and /r were removed while parsing below

'''

allPosWords = []
allNegWords = []

punctutations = string.punctuation[:14] + string.punctuation[15:]

for review in postivesD.values():
	print (review)
	review = review.translate(None, punctutations)
	print (review)
	allPosWords += review.split(" ")

posWordSet = set(allPosWords)
posWords = dict.fromkeys(posWordSet, 0)

for review in negativesD.values():
	allNegWords += review.split(" ")


# for review in negativesD.values():
# 	negWords[word] += 1

# print (posWords)


# print (len(postivesD), len(negativesD), len(positives), len(negatives))