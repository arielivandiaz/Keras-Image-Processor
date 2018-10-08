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
from nasnet import NASNet
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
	parser.add_argument('-ol', '--output_layer', type=str, default="0", help="the layer to get the output")
	#parser.add_argument('-fps', '--frame_rate', type=str, default=2, help="frames sampled per second")
	#parser.add_argument('-o', '--output_path', type=str,  default='processData', help="path for h5 outputfile")
	parser.add_argument('-pooling', '--pooling', type=str,  default='avg', help="pooling option avg / max")
	#parser.add_argument('-ext', '--file_extension', type=str,  default='mp4', help="the extension of video file, example: mp4")
	
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
	else:
		if args.model == 'inception':
			base_model = InceptionV3(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
			preprocess_mode='tf'
		elif args.model == 'xception':
			base_model = Xception(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
			preprocess_mode='tf'
		elif args.model == 'inceptionresnet':
			base_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
			preprocess_mode='tf'
		elif args.model == 'mobilenet':
			base_model = MobileNet(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
			preprocess_mode='tf'
		elif args.model == 'mobilenet2':	
			base_model = MobileNetV2(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
			preprocess_mode='tf'
		elif args.model == 'nasnet':	
			base_model = NASNetLarge(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
			preprocess_mode='tf'
		elif args.model == 'resnet':
			base_model = ResNet50(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
			preprocess_mode='caffe'
		elif args.model == 'vgg16':
			base_model = VGG16(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
			preprocess_mode='caffe'
		elif args.model == 'vgg19':
			base_model = VGG19(include_top=False, weights='imagenet', pooling=args.pooling)
			model = Model(input=base_model.input, output=base_model.get_layer(args.output_layer).output)
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
		return image.load_img(image_path, target_size=(224, 224))		
	elif args.model == 'resnet':
		return image.load_img(image_path, target_size=(331, 331))		
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
	print ("\n Predicting with %s ... \n" % args.model)


	model,preprocess_mode = load_model(args)
	img = load_image(img_path,args.model)

	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x,mode=preprocess_mode)

	features = []

	preds = model.predict(x)
	print('\nPredicted : ')


	print ("\n")

	if args.output_layer == '0':
		vector_len = preds[0].shape
		print ("Feature vector lenght : %d " % vector_len)
		#Save as a H5 File
		with h5py.File(img_path.split('.')[0] + '.h5',  "w") as file:
			
			#Model name as dataset title
			file.create_dataset(args.model,(1,),'i')	
			#Model name as ascii array
			asciiList = [n.encode("ascii", "ignore") for n in args.model]
			file.create_dataset('model_name',(len(asciiList),1),"|S3",asciiList)
			#Vector len
			file.create_dataset("vector_len",(1,),'i',vector_len)	
			#Features
			file.create_dataset("features",preds[0].shape,'f',preds[0])
			print ("H5 file saved")
	else:
		#If you want save a h5 file, you need analize the custom output layer shape
		#Maybe you need make a globalpolling for this layer
		vector_len = -1
		print ("Feature vector shape : \n \t\t\t\t")
		print (preds[0].shape)

	#print (preds[0])

