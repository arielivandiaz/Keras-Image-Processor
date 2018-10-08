from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from imagenet_utils import decode_predictions
from keras.preprocessing import image
#Models
from xception import Xception, preprocess_input

if __name__ == '__main__':

    model = Xception(include_top=True, weights='imagenet')

    img_path = 'kitten.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted : ')
    prediction = decode_predictions(preds, 5)
    for a in prediction[0]:
    	if len(a[1])<5:
    		print ("- %s \t\t\t : \t %d %% " % (a[1],a[2]*100))
    	else:
    		print ("- %s \t\t : \t %d %% " % (a[1],a[2]*100))
    	
