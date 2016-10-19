# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:57:11 2016

@author: Brandon
"""

import math
import numpy as np
from collections import Counter

# Maybe we should get rid of "the" "a" "and" etc. removed from bag of words

# This is a global variable used to map a bagofwords number to a word
# see load_vocabulary()
lookup = {}
# Global variable to map word to bagofwords number
# see load_vocabulary()
bagOfWords = {}

# These are the words we don't want to consider when classifying
badWords = ['.', '!', ',', '-', 'the', '"', 'a', 'and', 'of', \
                'for', 'in', ':', 'to', 'is', 'on', 'it', 'you', '&', 'it', 't', 'has', 'from', 'they', 'The']
#badWords = [],

def read_file(filename):
    """
    Spits out a file as a list
        INPUTS:     filename - path to the file that will be read
        RETURNS:    a list of string, where each element is a line in the file
                    WARNING: This includes newline characters
    """
    with open(filename, 'r') as f:
        return f.readlines()

def numberify(criminal):
    """
    Given a criminal, gives his/her (theoretical) prisoner number
        INPUTS:     criminal - string containing the name of the criminal
        RETURNS:    the prisoner number (class) of the criminal (label)
    """
    if criminal == "HillaryClinton":
        return 0
    elif criminal == "realDonaldTrump":
        return 1
    else:
        print "ERROR: Criminal not detected"
        return 5097     # MissingNo.

def load_labels():
    """
    Spits out two lists, one for training, one for dev.
    Each list contains the class corresponding to each example
    """
    tempLabels =    read_file('clintontrump.labels.train')
    labelsTrain =   [numberify(a.strip()) for a in tempLabels]
    tempLabels =    read_file('clintontrump.labels.dev')
    labelsDev =     [numberify(a.strip()) for a in tempLabels]
    return labelsTrain, labelsDev

def load_data():
    """
    Reads from the bagofwords files.
    IF YOU DON'T HAVE THESE, RUN THE SCRIPT FROM JUNKI (The TA)
    ATTENTION: these contain newlines, but don't think it'll be an issue
    see read_file() description
    """
    dataTrain =      read_file('clintontrump.bagofwords.train')
    dataDev =        read_file('clintontrump.bagofwords.dev')
    dataTest =       read_file('clintontrump.bagofwords.test')
    load_vocabulary() # IMPORTANT
    
    return dataTrain, dataDev, dataTest

def load_vocabulary():
    """
    Creates a python dict for mapping from numbers to words
    """
    global lookup
    global bagOfWords # Let's us modify global var 
    lookup.clear()    
    
    vocab =         read_file('clintontrump.vocabulary')

    for line in vocab:
        pair = line.split()
        number = pair[0]; word = pair[1].strip()
        lookup[number] = word
        bagOfWords[word] = number
    return
    
def train_classifier(data, labels, numClasses, multinomial, use_Laplace, alpha=None):
    """
    Trains our classifier (duh); basically does a bunch of magic
    This is kinda a general function that can be used for other Naive Bayes projects
        INPUTS:     data -       list of examples containing features, which 
                                 are contained in a string (bagofwords)
                    labels -     list of classes, matches examples in data
                    numClasses - how many classes are we dealing with?
                    multinomial -Use multinomial model? (defaults to Bernoulli)
                    k -          Laplace smoothing parameter 
                                 (most likely only takes on values of 0 or 1;
                                 0 means off, 1 means on)
        RETURNS:    dics -       a list of dictionaries; each class has it's
                                 own dictionary
                                 key: a word
                                 value: its probability of appearing in an 
                                        example of class
                    priors -     each class's prior probability
                    classCount - how many times a given class appeared
                                 (Not sure if necessary)
    """
    if len(data) != len(labels):
        print "ERROR: does the data match the labels?"    
        return
    
    # Initialize necessary stuff
    
    # So apparently you can't do [{}]*3 to make a list of 3 dictionaries
    # Well, you'll make 3, but they will all reference the same thing
    # so if you add a key to one dictionary it gets added to the others
    # F****n' weird, but alright... Let this be a lesson
    #dics = [{}] * numClasses
    dics = []  # [List] of {dictionaries}
    for c in range(numClasses):
        dics.append({}) 
    classCount = [0.0] * numClasses
    priors = [0.0] * numClasses
    wordCount = [0.0] * numClasses # total # of words counted for each class
    for e in range(len(data)):  # e means example number
        # The eth example
        example = data[e]
        # Remember which class this belongs to        
        c = labels[e]
        # Add one to the class's count
        classCount[c] += 1.0
        
        words = example.split() # Separate example into words
        wordCount[c] += len(words) # update total # of words for class
        if not multinomial: # Bernoulli
            words = set(words) # remove multiples 
        for word in words:  # Iterate through words
            word = lookup[word.strip()] # Find the ACTUAL word from the number
            if word in badWords:   # don't add common words or punctuation
                break
            # pick the dictionary of matching the example's class (lik is for likelihood)
            lik = dics[c] # THIS IS A REFERENCE, so it modifies dics accordingly
            if word in lik: # Did we see this word (in this class) already?
                lik[word] += 1.0 # +1 to number of times the word has appeared
            else:
                lik[word] = 1.0 # Add the word to the dictionary

    # Now we need to do some maths
    for c in range(numClasses): # iterate through classes
        lik = dics[c] # Again, lik references part of dics
        for word in lik: # for every word in our dictionary
            # find it's probability of appearing in a "document" of class c
            if multinomial:
                if use_Laplace:
                    lik[word] = (lik[word] + alpha) / (wordCount[c] + alpha * len(bagOfWords))
                else:
                    lik[word] /= wordCount[c]
            else: # Bernoulli
                lik[word] /= classCount[c] 

            """ !LAPLACE SMOOTHING MOST LIKELY GOES HERE! """
            # below is my guess for how it would work (you would replace lines above)
            # lik[word] = (lik[word] + k) / (classCount[c] + k * 2)  [BERNOULLI]
            # lik[word] = (lik[word] + k) / (wordCount[c] + k * len(lik)) [MULTINOMIAL]
            #                                                        ^
            # The slides say that |D| is the size of the dictionary, but I don't
            # know if it's the size of dictionary for ALL classes or just class c
            # And for 

        # Calculate priors for class c
        priors[c] = classCount[c]/len(data)

        # if multinomial and use_Laplace: #Laplace smoothing for priors
        #     priors[c] = (classCount[c] + alpha*classCount[c]) / (len(data) + len(data) * alpha * 2)

    
    return dics, priors, classCount
    
def test_classifier(data, dics, priors, multinomial = False):
    """
    After training, use this function to test classifier on new set of data.
        INPUTS:     data -      list of examples containing features, which 
                                are contained in a string (bagofwords)
                    dics -      a list of dictionaries, with words and their probs
                                captured from training; each class has it's
                                own dictionary
                                key: a word
                                value: its probability of appearing in an 
                                    example of class
                    priors -    the prior probs captured from the train data
                    multinomial-Use multinomial model? (defaults to Bernoulli)
        RETURNS:    predictions-the classifiers predicted classifications
                                in order of the examples presented
    """
    predictions = []
    numClasses = len(dics) # hehe
    for example in data: # Iterate through examples
        posteriors = np.array([0.0]*numClasses) # this will be P(y|x)
        for c in range(numClasses): # go through each class
            posteriors[c] = math.log(priors[c]) # add log of prior prob
            # Little Lindsey Likelihood
            lik = dics[c]
            words = example.split() # get bagofwords numbers in a list
            for w in range(len(words)): # convert numbers to actual words
                words[w] = lookup[words[w].strip()]
            # here's the juicy part, it's is the capital pi product thing in the P(y|x) = ...
            for myWord in lik: # iterate through each word
                p = 0.0
                if multinomial:
                    if myWord in words:
                        xi = Counter(words)[myWord] # This counts occurances of myWord in the tweet
                        # Second equation in implementation instructions, but with log
                        p = xi*math.log(lik[myWord])
                else: # Bernoulli
                    xi = myWord in words # xi is either 1 or 0
                    # First equation in implementation instructions, but with log
                    p = xi*math.log(lik[myWord]) + (1 - xi)*math.log(1 - lik[myWord])
                posteriors[c] -= p # "multiply" probability to calc P(y|x)
        predictions.append(np.argmax(posteriors))
    return predictions
    
def get_test_accuracy(labels, predictions):
    """
    Given true classifications and predictions, calculate the accuracy of predictions
    """
    if len(labels) != len(predictions):
        print "ERROR: you must have plugged in something wrong. Can't calculate accuracy"
        return
    
    correct = 0.0
    for e in range(len(labels)):
        if labels[e] == predictions[e]:
            correct += 1
    
    return correct / len(labels)
     
def most_likely_words(dics):
    """
    Get's the 10 most likely words from each classs
        INPUTS:     dics -       a list of dictionaries, with words and their probs
                                 captured from training; each class has it's
                                 own dictionary
                                 key: a word
                                 value: its probability of appearing in an 
                                    example of class
        RETURNS:    mostLikely - list of lists of most likely words
                                 ith list is words for class i
    """
    mostLikely = []
    for lik in dics: # iterate through class dictionaries
        topWords = [] # the words with highest likelihood for class
        # creates numpy array where first row has bagofwords number
        # second row has probability of word appearing        
        bowToLik = np.array([(float(bagOfWords[word]), float(lik[word])) for word in lik]).T
        for m in range(10): # find the 10 words
            idxMax = np.argmax(bowToLik[1]) # find the highest prob, record column index
            # use index to get bagofwords number, then convert to actual word and add to list        
            topWords.append(lookup[str(int(bowToLik[0][idxMax]))])
            # delete the column so we can find the next max
            bowToLik = np.delete(bowToLik, [idxMax], axis=1)
        # add list of highest prob words for class
        mostLikely.append(topWords)
    return mostLikely

def confusion_matrix(labels, predictions, numClasses):
    #         table looks like this
    #        ---------------------------------
    #        |           | class 0 | class 1 |
    #        ---------------------------------
    #        | predict 0 |         |         |
    #        ---------------------------------
    #        | predict 1 |         |         |
    #        ---------------------------------
    if len(labels) != len(predictions):
        print "ERROR: you must have plugged in something wrong. Can't calculate confusion matrix"
        return

    confusion = np.array([[0.0] * numClasses] * numClasses)
    for truth, pred in zip(labels, predictions):
        confusion[pred][truth] += 1.0        
    
    confusion /= len(labels)
    
    return confusion









