# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 14:47:08 2016

@author: Brandon
"""
use_Laplace = 1
alpha = 1
multinomial = 1
import impl2_funcs as funcs, sys

# y = 0 -> Clinton tweet
# y = 1 -> Trump tweet

################## TRAINING PROCEDURE ########################################
# NOTE: order doesn't actually matter for steps 1 and 2
# 1. Find how many tweets each criminal tweeted
#   This determines P(y) - ex: P(y=0) = # of Clinton tweets/# total tweets
#   P(y) is called the prior
# 2. Find P(xi|y), ex: P(xi=1|y=1) is prob of ith word appearing in a Trump tweet
#
#                    # of times ith word is present in Trump tweets
#      P(xi=1|y=1) = ----------------------------------------------
#                              total # of Trump tweets

labelsTrain, labelsDev = funcs.load_labels()
dataTrain, dataDev, dataTest = funcs.load_data()



dics, priors, count = funcs.train_classifier(dataTrain, labelsTrain, 2, multinomial, use_Laplace, alpha)
mostLikely = funcs.most_likely_words(dics)



# predictions = funcs.test_classifier(dataTrain, dics, priors, multinomial)
# print mostLikely
# print funcs.get_test_accuracy(labelsTrain, predictions)
# print funcs.confusion_matrix(labelsTrain, predictions, 2)
# sys.exit()

predictions = funcs.test_classifier(dataDev, dics, priors, multinomial)
#print mostLikely
print funcs.get_test_accuracy(labelsDev, predictions) * 100
#print funcs.confusion_matrix(labelsDev, predictions, 2)