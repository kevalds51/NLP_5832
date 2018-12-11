# NLP_5832

Coursework repository for Natural Language Processing course

     ---------------------------------------- *** Assignment 1 *** ----------------------------------------

How many words? How many words do you know in your native language?

Provide an answer and a short rationale for it (no longer than necessary).  Your rationale should include some discussion of what you decided what a word was, what "knowing a word" meant, what "how many" meant,  and the method you used to come up with the estimate.


     ---------------------------------------- *** Assignment 2 *** ----------------------------------------

In this assignment, we'll explore the statistical approach to part-of-speech tagging.

Probabilistic Taggers

A. Baseline

As a baseline system, you should first implement a "most frequent tag" system. Given counts from the training data, your tagger should simply assign to each input word the tag that it was most frequently assigned to in the training data.  This is a reasonable approach and will allow you to assess the performance of your more advanced approach.

B. Viterbi

Once you have a working baseline, implement the Viterbi algorithm with a bigram-based approach. Specifically, you'll need to:

   	1. Extract the required counts from the training data to generate the required probability estimates for the model.
    2. Deal with unknown words in some sensible way.
    3. Do some form of smoothing for the bigram tag model.
    4. Implement a Viterbi decoder.
    5. Evaluate your system's performance on unseen data.

     ---------------------------------------- *** Assignment 3 *** ----------------------------------------

In this assignment, you will implement a text classification system for sentiment analysis. I will provide training data in the form of positive and negative reviews. Use this training data to develop a system that takes reviews as test cases and categorizes them as positive or negative. 

     ---------------------------------------- *** Assignment 4 *** ----------------------------------------

In this assignment, you are to implement a learning-based approach to named entity recognition.  In this approach, we can cast the problem of finding named entities as a sequence labeling task using IOB tags. One possible framework is to use the HMM-based solution developed for your POS tagging assignment.  The particular NER task we’ll be tackling is to find all the references to genes in a set of biomedical journal article abstracts. 