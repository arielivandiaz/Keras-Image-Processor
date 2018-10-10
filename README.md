# Keras-Image-Processor

## [YouTube Demo:](http://www.youtube.com/watch?v=vzpp1dO4i44 "Video Demo link")
[![Video Demo](https://img.youtube.com/vi/vzpp1dO4i44/0.jpg)](http://www.youtube.com/watch?v=vzpp1dO4i44)



### Sample Files

  - **demoy.py :** didactic example used to make the video demo
  - **sample_predict.py : ** predict keywords from an image
  - **sample_video_predict.py : ** predict a sequence of keywords from  video frames
  - **sample_feature_extractor.py : ** extract the features vector of an image from a model intermediate layer (the last preferably)
  - **sample_feature_extractor_all_models.py : **   extract the features using all models
  - **sample_video_feature_extractor.py : ** extract the features from a video with a sample rate, and save it how a h5 matrix


## Models:

  - **inception_resnet_v2.py: **
  - **inception_v3.py : **
  - **mobilenet.py : **
  - **mobilenet_v2.py : **
  - **nasnet.py : **
  - **resnet50.py : **
  - **vgg16.py : **
  - **vgg19.py : **
  - **xception.py : **

## Arguments

  - **-i** (args.input) : path to input image/video
  - **-m** (args.model) : select the model:  inception, resnet, vgg16, vgg19, xception, etc
  - **-ol** (args.output_layer)  the layer to get the output, you need know the name and process the structure
  - **-fps** (args.frame_rate) : frames sampled per second, default=2
  - **-o** (args.output_h5): path/filename for h5 output file
  - **-pooling** (args.pooling) : model pooling option avg / max


