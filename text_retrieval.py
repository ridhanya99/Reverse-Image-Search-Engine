#print("\nText_retrieval Loaded...")

import string
import numpy as np

import os
from pickle import dump, load
from numpy import argmax

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences

from nltk.translate.bleu_score import corpus_bleu
from IPython.display import Image, display
from shutil import copyfile

#--------------------required functions.............................

def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# Function for loading a pre-defined list of photo identifiers
def load_photo_identifiers(filename):
    file = load_file(filename)
    photos = list()
    for line in file.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        photos.append(identifier)
    return set(photos)

def load_clean_descriptions(filename, photos):
    file = load_file(filename)
    descriptions = dict()
    for line in file.split('\n'):
        words = line.split()
        image_id, image_description = words[0], words[1:]
        if image_id in photos:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = ' '.join(image_description)
            descriptions[image_id].append(desc)
            
    return descriptions

#----------------------

def load_description_mapping(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line)<2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping    

#--------------------------------------

dataset_root_dir = "D:/XAMPP/htdocs/Package/Image_Retrieval/"

input_file = open('D:/XAMPP/htdocs/uploads/description.txt', 'r')
predicted_description = input_file.readline()


table = str.maketrans('','',string.punctuation)
desc = predicted_description.split()
desc = [word.lower() for word in desc]
desc = [word.translate(table) for word in desc]
desc = [word for word in desc if len(word)>1]
desc = [word for word in desc if word.isalpha()]
predicted_description =  ' '.join(desc)

#---------

photo_file = dataset_root_dir + 'images.txt'
photoLabel = load_photo_identifiers(photo_file)
photo_descriptions = load_clean_descriptions(dataset_root_dir+ '/image_desc.txt', photoLabel)


matchedFiles = set()

for img in photoLabel:
    actual, predicted = list(), list()
    yhat = predicted_description.split()
    predicted.append(yhat)
    references = [d.split() for d in photo_descriptions[img]]
    actual.append(references) 
    bleu_score_1 = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0))
    bleu_score_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_score_3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.34, 0))
    bleu_score_4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    bleu_score = ( 4*bleu_score_4 + 3*bleu_score_3 + 2*bleu_score_2 + bleu_score_1 )/10
    
    if bleu_score > 0.2:
        #print(bleu_score)
        matchedFiles.add(img)
        continue

#--------
folder = 'D:/XAMPP/htdocs/uploads/matched-images'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

#------        

matched_img_file = open('D:/XAMPP/htdocs/uploads/matched_images.txt',"w")
img_root = dataset_root_dir+'Images/'

desc_text = load_file(dataset_root_dir + 'image_desc.txt')
descriptions = load_description_mapping(desc_text)

i=0
for img in matchedFiles:
    img_path = img_root + img + '.jpg'
    i += 1
    matched_img_file.write(descriptions[img][0]+ '\n')
    copyfile(img_path, folder + '/' + format(i,'03d') + '.jpg')
    
matched_img_file.close()