# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import os
import cv2
import torch
import time


batch_size = 1 #Numero de muestras para cada batch (grupo de entrada)

def load_test(data_path):
	X_test = []
	images_names = []
	image_path = data_path
	print('Read test images')
	
	for imagen in [imagen for imagen in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, imagen))]:
		imagenes = os.path.join(image_path, imagen)
		#print(imagenes)
		img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (224, 224))
		X_test.append(img)
		images_names.append(imagenes)
	return X_test, images_names

def read_and_normalize_test_data(data_path):
    test_data, images_names = load_test(data_path)
    test_data = np.array(test_data, copy=False, dtype=np.float32)
    return test_data, images_names

def normalize_test_data(images):
	images_clone = images[0].clone().numpy()
	image_return = []
	for i in range(len(images_clone)):
		a = images_clone[i]
		a = a.copy().transpose(1,2,0)
		a = np.float32(cv2.resize(a, (224, 224)))
		a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
		a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
		#a = np.array(a, copy=False, dtype=np.float32)
		image_return.append(a)
	return image_return


def GetDatasetScores(data_path, model_name = 'FaceQnet_v1.h5', save_name = 'score_qualities.txt', model_dir = 'models/'):
		

  #Loading one of the the pretrained models
  
  # model = load_model('FaceQnet.h5')
  
  model = load_model(model_dir + model_name)
  
  #See the details (layers) of FaceQnet
  # print(model.summary())
  
  #Loading the test data
  test_data, images_names = read_and_normalize_test_data(data_path)
  y=test_data
  
  #Extract quality scores for the samples
  score = model.predict(y, batch_size=batch_size, verbose=0)
  predictions = score
  
  
  #Saving the quality measures for the test images
  fichero_scores = open(save_name,'w')
  i=0
  
  
  #Saving the scores in a file
  fichero_scores.write("img;score\n")
  for item in predictions:
  	fichero_scores.write("%s" % images_names[i])
  	#Constraining the output scores to the 0-1 range
  	#0 means worst quality, 1 means best quality
  	if float(predictions[i])<0:
  		predictions[i]='0'
  	elif float(predictions[i])>1:
  		predictions[i]='1'
  	fichero_scores.write(";%s\n" % predictions[i])
  	i=i+1

def GetImagesScores(images, model_name = 'FaceQnet_v1.h5', model_dir = 'models/'):

	st = time.time()

	model = load_model(model_dir + model_name)
	#See the details (layers) of FaceQnet
	# print(model.summary())
	#Loading the test data
	test_data = normalize_test_data(images)
	y= np.array(test_data)
	cv2.imwrite("res5/teste.jpg",y[2])
	#Extract quality scores for the samples
	score = model.predict(y, batch_size=1, verbose=0)

	score = np.array(score).squeeze()
	#Saving the quality measures for the test images
	return score

def GetDataloaderScores(dataloader, model_name = 'FaceQnet_v1.h5', save_name = 'score_qualities.txt', model_dir = 'models/'):
	preds = []
	for imgs, img_nm, imgl, bb in dataloader:
		imgs[:][0]

		scores = GetImagesScores(images = imgs, model_name = model_name, model_dir = model_dir)
		preds.extend(scores)
	return preds
if __name__ == '__main__':

	GetImagesScores(data_path = '../datasets/LFW/lfw/Zico')
