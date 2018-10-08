from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import argparse
import os
import cv2
from PIL import Image
import operator

from imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing import image

#Models
from xception import Xception
from inception_v3 import InceptionV3
from inception_resnet_v2 import InceptionResNetV2
from mobilenet import MobileNet
from mobilenet_v2 import MobileNetV2
from nasnet import NASNetLarge
from resnet50 import ResNet50
from vgg16 import VGG16
from vgg19 import VGG19


#Demo stuff
from cool_stuff import *


def read_args():

	parser = argparse.ArgumentParser("Extract features and prediction from every images")
	parser.add_argument('-i', '--input', type=str, default="video.mp4", help="path to input image")
	parser.add_argument('-m', '--model', type=str, default="inception", help="select the model:  inception, resnet, vgg16, vgg19, xception, etc")
	parser.add_argument('-ol', '--output_layer', type=str, default="0", help="the layer to get the output")
	parser.add_argument('-fps', '--frame_rate', type=str, default=2, help="frames sampled per second")
	parser.add_argument('-o', '--output_h5', type=str,  default='models_features', help="path for h5 outputfile")

	
	
	return parser.parse_args()

def load_model (args):

	if args.model == 'inception':
		model = InceptionV3(include_top=True, weights='imagenet')
		preprocess_mode='tf'
	elif args.model == 'xception':
		model = Xception(include_top=True, weights='imagenet')
		preprocess_mode='tf'
	elif args.model == 'inceptionresnet':
		model = InceptionResNetV2(include_top=True, weights='imagenet')
		preprocess_mode='tf'
	elif args.model == 'mobilenet':
		model = MobileNet(include_top=True, weights='imagenet')
		preprocess_mode='tf'
	elif args.model == 'mobilenet2':	
		model = MobileNetV2(include_top=True, weights='imagenet')
		preprocess_mode='tf'
	elif args.model == 'nasnet':	
		model = NASNetLarge(include_top=True, weights='imagenet')
		preprocess_mode='tf'
	elif args.model == 'resnet':
		model = ResNet50(include_top=True, weights='imagenet')
		preprocess_mode='caffe'
	elif args.model == 'vgg16':
		model = VGG16(include_top=True, weights='imagenet')
		preprocess_mode='caffe'
	elif args.model == 'vgg19':
		model = VGG19(include_top=True, weights='imagenet')
		preprocess_mode='caffe'
	else:
		print ("Model not found")

	return model,preprocess_mode


def load_image(frame, model_name):

	if args.model == 'inception':
		img = Image.fromarray(frame, 'RGB')
		img = img.resize((299, 299))	
		return np.array(img)
		return image.load_img(image_path, target_size=(299, 299))
	elif args.model == 'xception':
		return image.load_img(image_path, target_size=(299, 299))		
	elif args.model == 'inceptionresnet':
		return image.load_img(image_path, target_size=(299, 299))		
	elif args.model == 'mobilenet':
		return image.load_img(image_path, target_size=(224, 224))	
	elif args.model == 'mobilenet2':
		return image.load_img(image_path, target_size=(224, 224))		
	elif args.model == 'nasnet':
		return image.load_img(image_path, target_size=(331, 331))		
	elif args.model == 'resnet':
		return image.load_img(image_path, target_size=(224, 224))		
	elif args.model == 'vgg16':
		return image.load_img(image_path, target_size=(224, 224))
	elif args.model == 'vgg19':
		return image.load_img(image_path, target_size=(224, 224))
	else:
		print ("Model not found")

	return  img


if __name__ == '__main__':

	args = read_args()

	
	print ("\n Predicting with %s ... \n" % args.model)

	vidcap = cv2.VideoCapture(args.input)
	total_frames = 	vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
	fps=vidcap.get(cv2.cv2.CAP_PROP_FPS)
	offset = np.rint(fps)/ args.frame_rate	
	success,frame = vidcap.read()
	frame_count = 0

	model,preprocess_mode = load_model(args)

	prediction_dict = {}

	while success:

		img = load_image(frame,args.model)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x,mode=preprocess_mode)
		preds = model.predict(x)
	
		prediction = decode_predictions(preds, 3)

		vidcap.set(1,frame_count)
		success,frame = vidcap.read()
		frame_count += offset

		#Print frame prediction
		#print ("Frame : %d -> \t %s , \t %s , \t %s" % (frame_count,prediction[0][0][1],prediction[0][1][1],prediction[0][2][1]))

		#Make a dictionary with probs
		for i in range(0,3):			
			if str(prediction[0][i][1]) in  prediction_dict :
				prediction_dict[str(prediction[0][i][1])]+=prediction[0][i][2]
			else : 
				prediction_dict[str(prediction[0][i][1])]=prediction[0][i][2]
		

		if success  :
			cv2.imshow("Video Predict", frame)
			cv2.waitKey(10)

		

	print ("\n********************** Results ******************* \n")
	sorted_prediction_dict =  sorted(prediction_dict.items(), key=operator.itemgetter(1), reverse=True)

	for a in sorted_prediction_dict:
		print("\t keyword : %s \t - score : %f " % (a[0],a[1]*100))

	print ("\n")
	cv2.destroyAllWindows()
	#searh prediction in google images:	
	#open_google_image(prediction[0][0][1])
	#display_image(args.input)
