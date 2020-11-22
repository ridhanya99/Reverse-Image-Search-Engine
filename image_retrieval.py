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

def extract_features(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

#----------------------------

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

# function to load the photo features created using the VGG16 model
def load_photo_features(filename, photos):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in photos}
    
    return features

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


#-------------------------------------Get matched images to the querry...................................
root_dir = 'D:/XAMPP/htdocs/Package/Caption_Generator/'

tokenizer = load(open(root_dir+'tokenizer.pkl', 'rb'))
max_length = 34

model = load_model(root_dir+'model.h5')

img = "D:/XAMPP/htdocs/uploads/photo.jpg"
photo = extract_features(img)
predicted_description = generate_desc(model, tokenizer, photo, max_length)
print_description = ' '.join(predicted_description.split(' ')[1:-1])

desc_file = open('D:/XAMPP/htdocs/uploads/description.txt',"w")
desc_file.write(print_description)
desc_file.close()

#--------

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
dataset_root_dir = 'D:/XAMPP/htdocs/Package/Image_Retrieval/'

photoFile = dataset_root_dir +'images.txt'
photoImagesLabel = load_photo_identifiers(photoFile)
photo_descriptions = load_clean_descriptions(dataset_root_dir+'image_desc.txt', photoImagesLabel)

matchedFiles = set()

for img in photoImagesLabel:
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
