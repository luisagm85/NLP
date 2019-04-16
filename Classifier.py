# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:43:47 2019
#https://rpubs.com/Joaquin_AR/334526
@author: lgomez
"""
import pandas as pd
import xml.etree.ElementTree as et 
import os 
from os import walk, path
from os.path import join
import re #regex
import numpy as np
from nltk.tokenize import TweetTokenizer

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

hashtag_pattern = re.compile(r"#(\w+)")

htmlCharDict = {'&amp;': '&', '&gt;': '>', '&lt;': '<', '&quot;': '\"'}

stopWords = ["a", "an", "and", "are", "as", "at",
                "be", "but", "by",
                "de", "des",
                "e", "el",
                "for",
                "hi", "hey", "he",
                "if", "i", "in", "into", "is", "it", "it's", "its",
                "la", "le", "les",
                "no", "not",
                "of", "on", "or",
                "que",
                "row",
                "s", "she", "so", "such",
                "to", "that", "the", "their", "then", "there", "these", "they", "they're", "this",
                "was", "where", "will", "with"]

#Método  cargar datos entrenamiento
def readTrainingData():
    #Cargue archivo TXT con los IDs de los Tweets para entrenamiento
    train_ids = pd.read_csv('./en/truth-train.txt', sep=':::', engine='python', header=None)
    train_ids.columns = ["tweetID", "type", "gender"]
    
    #Declarar DataFrame vacio para almacenar la información de entrenamiento
    train_data = pd.DataFrame({'tweetID':[],'text':[],'type':[],'gender':[]})
    
    #Cargar los tweets de training en un dataset
    for row in train_ids.itertuples():        
        #Buscar el archivo XML y cargar el texto en el DataFrame de training
        path = row.tweetID + ".xml"
        doc = et.parse("./en/" + path)
        root = doc.getroot()
        for tweet in root.iter('document'):
            text = tweet.text
            train_data.loc[len(train_data)]=[row.tweetID,text,row.type,row.gender]
    
    clean_train_data = cleanData(train_data)    
    print(clean_train_data['text'])

 
def cleanData(data):
    
    clean_df = data.copy()
    
    # Forcing all string characters to be lower case.
    clean_df['text'] = clean_df['text'].str.lower()
    
    # Removing html tags and attributes.
    htmlTagsRgx = r'<[^>]+>'
    clean_df['text'] = clean_df['text'].str.replace(htmlTagsRgx, '')
    
    # Replacing html character codes.    
    for key, value in htmlCharDict.items():
        clean_df['text'] = clean_df['text'].str.replace(key, htmlCharDict[key])
        
    # Replacing literal \' to literal apostrophes.
    litAposRgx = r'\\+\''
    clean_df['text'] = clean_df['text'].str.replace(litAposRgx, '\'')
    
    # Removing all urls.
    urlRgx = r'(?:\S+(?=\.[a-zA-Z])\S+)'
    clean_df['text'] = clean_df['text'].str.replace(urlRgx, '')
    
    # Removing RT tags.
    clean_df['text'] = clean_df['text'].str.replace("rt", '')
    
    # Removing Emojis (OPTIONAL)
    clean_df['text'] = clean_df['text'].str.replace(emoji_pattern, 'emoji')
    
    #Removing Hashtags
    clean_df['text'] = clean_df['text'].str.replace(hashtag_pattern, '')
    
    #Removing stopwords
    stopRemove = '|'.join(stopWords)
    stopRgx = r'\b('+stopRemove+r')\b'
    clean_df['text'] = clean_df['text'].str.replace(stopRgx, '')

    # Removing punctuation (OPTIONAL)
    punctRgx = r'[^\w\s]'
    clean_df['text'] = clean_df['text'].str.replace(punctRgx, '')

    # Removing twitter handles.
    handle_rgx = r'@\S+'
    clean_df['text'] = clean_df['text'].str.replace(handle_rgx, '')
    
    tknzr = TweetTokenizer()
    for tweet in clean_df['text'].value
        tknzr.tokenize(tweet)

    return clean_df
    
if __name__ == "__main__":   
    readTrainingData()
    #print(clean_train_data['text'])
