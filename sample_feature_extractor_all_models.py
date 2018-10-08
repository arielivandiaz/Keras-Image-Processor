from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import argparse
import os
import h5py

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

from keras.models import Model

#Demo stuff
from cool_stuff import *


def read_args():

	parser = argparse.ArgumentParser("Extract features and prediction from every images")
	parser.add_argument('-i', '--input', type=str, default="kitten.jpg", help="path to input image")
	parser.add_argument('-m', '--model', type=str, default="xception", help="select the model:  inception, resnet, vgg16, vgg19, xception, etc")
	parser.add_argument('-o', '--output_h5', type=str,  default='models_features', help="path for h5 outputfile")
	parser.add_argument('-pooling', '--pooling', type=str,  default='avg', help="pooling option avg / max")
	
	
	return parser.parse_args()

def load_model (args):

	if args.output_layer == '0':
		if args.model == 'inception':
			model = InceptionV3(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='tf'
		elif args.model == 'xception':
			model = Xception(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='tf'
		elif args.model == 'inceptionresnet':
			model = InceptionResNetV2(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='tf'
		elif args.model == 'mobilenet':
			model = MobileNet(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='tf'
		elif args.model == 'mobilenet2':	
			model = MobileNetV2(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='tf'
		elif args.model == 'nasnet':	
			model = NASNetLarge(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='tf'
		elif args.model == 'resnet':
			model = ResNet50(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='caffe'
		elif args.model == 'vgg16':
			model = VGG16(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='caffe'
		elif args.model == 'vgg19':
			model = VGG19(include_top=False, weights='imagenet', pooling=args.pooling)
			preprocess_mode='caffe'
		else:
			print ("Model not found")
			return 0


	return model,preprocess_mode


def load_image(image_path, model_name):

	if args.model == 'inception':
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

	img_path = args.input

	models = 	['inception','xception','inceptionresnet','mobilenet','mobilenet2','resnet','vgg16','vgg19']

	file = h5py.File(args.output_h5 + '.h5',  "w")
	if not file:
		print ("Unable to create h5 output file")

	for model_name in models:

		args.model=model_name
		print ("\n Predicting with %s ... \n" % args.model)

		model,preprocess_mode = load_model(args)
		img = load_image(img_path,args.model)

		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x,mode=preprocess_mode)
		preds = model.predict(x)

		print('\nPredicted : ')
		vector_len = preds[0].shape
		print ("Feature vector lenght : %d " % vector_len)

		#Save as a H5 File		
		#Create a group with the model name
		group = file.create_group(model_name)		
		#Model name as ascii array
		asciiList = [n.encode("ascii", "ignore") for n in args.model]
		group.create_dataset('model_name',(len(asciiList),1),"|S3",asciiList)
		#Vector len
		group.create_dataset("vector_len",(1,),'i',vector_len)	
		#Features
		group.create_dataset("features",preds[0].shape,'f',preds[0])
	
	print ("H5 file saved")
	print ("Succes")