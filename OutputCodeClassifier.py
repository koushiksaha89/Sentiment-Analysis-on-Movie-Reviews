# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:52:56 2015

@author: Koushik Saha
"""

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

#Path to read from
data_path=os.path.abspath('data')
train_file=os.path.join(data_path,'train.tsv')
test_file=os.path.join(data_path,'test.tsv')

#Read the files from the data folder
print "Reading Training and Test Data: \n"
train=pd.read_csv(train_file,delimiter="\t",header=0)
test=pd.read_csv(test_file,delimiter="\t",header=0)

##shapes of the file
print "Train file shape:  ", train.shape
print "Test file shape:    ", test.shape

print "\nFitting pipeline.. \n"

#Bag of Words from Sklearn
print "Running Feature Extraction.."
vectorizer=CountVectorizer() #initialise Bag of words
train_count=vectorizer.fit_transform(train.Phrase)
print "Bag of words Counts: ", train_count.shape

#Tf-Idf Transformer
print "Running Tf-Idf Transformer"
tf_idf=TfidfTransformer() #initialise Tf-Idf Transformer
train_tf_idf=tf_idf.fit_transform(train_count)
print "Tf-Idf : ", train_tf_idf.shape

#Process the test set
print "Processing Test set.. \n"
test_count=vectorizer.transform(test.Phrase)
test_tf_idf=tf_idf.transform(test_count)

print "Training the Model and predicting on the Test data.."
predicted1=OutputCodeClassifier(LinearSVC(random_state=0),code_size=2,random_state=0).fit(train_tf_idf, train.Sentiment).predict(test_tf_idf)


print "Writing the output in a csv file..."
output=pd.DataFrame(data={"PhraseId":test.PhraseId,"Sentiment":predicted1})
output.to_csv("Sentiment Analysis on Movie Reviews -- OutputCode",index=False,quoting=3)