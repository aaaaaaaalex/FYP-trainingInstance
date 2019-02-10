from tensorflow import logging
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np
import argparse

# silence tensorflow's useless info logging
logging.set_verbosity(logging.ERROR)

# parse arguments (image file's path)
argparser = argparse.ArgumentParser()
argparser.add_argument('file')
args = argparser.parse_args()

# get testing image from test_src and VGG16 model from keras
test_src =  args.file
test_image = image.load_img(test_src, target_size=(224,224))
base_model = VGG16(include_top=True, weights="imagenet")

# convert image to numerical array and reshape dimensions to match vgg input (224, 224, 3)
x_input = image.img_to_array(test_image)
x_input = np.expand_dims(x_input, axis=0)
x_input = preprocess_input(x_input)
pred    = decode_predictions( base_model.predict(x_input), top=3 )[0]


# parallel arrays for simplicity
classes = []
datapoints = []
for cls in pred:
    classes.append( cls[1] )
    datapoints.append( cls[2] )
item_index = np.arange(len(classes))

print("classes: {}, datapoints: {}".format(classes, datapoints))

# show image tested and corresponding machine predictions
image_figure = plt.figure(1)
plt.imshow(test_image)
image_figure.show()

prediction_figure = plt.figure(2)
plt.xticks( item_index , classes)
plt.ylabel('certainty')
plt.bar(item_index, datapoints, align='center')
prediction_figure.show()

#pause execution until input is received
input()






