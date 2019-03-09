from os import listdir, makedirs, path
from io import BytesIO

from keras.applications.densenet import DenseNet201, preprocess_input, decode_predictions
from keras.preprocessing import image as kimage
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from tensorflow import logging
from PIL import Image as pimage
from matplotlib import pyplot as plt

import argparse
import requests
import numpy as np
import pandas as pd

# attempts to validate if a collection of bytes represents an image by parsing them with PIL.Image
# throws an exception if the data cannot be parsed
def validate_image(bytes_data):
    buf = BytesIO(bytes_data)
    img = pimage.open(buf)


def download_imgs(links, save_dir, download_limit=100):
    # throw error if links isnt a list
    assert type(links) is list
    images_pulled = []
    
    # iterate over each link, carrying both the link and it's list index
    i=0
    for link, i in zip(links, range(len(links))):
        if i > download_limit: break 
        try:
            # make a GET request and dont follow redirects - timeout after 3 secs
            r = requests.get(link, allow_redirects=False, timeout=3)
            r.raise_for_status()

            # make sure the response is an image, not HTML
            validate_image(r.content)
            filename = "{}img-{}.jpg".format(save_dir, i)
            if not path.exists(save_dir):
                makedirs(save_dir)
            with open(filename, 'wb') as f:
                f.write(r.content)
            images_pulled.append(filename)
        except (requests.exceptions.RequestException, OSError):
            pass
    
    return images_pulled


def get_training_classes(classfile='./dataset/classes.config'):
    classes = []
    with open(classfile) as fd:
        classes = [line.rstrip('\n') for line in fd]         
    return classes


def read_links(src):
    links = []
    with open(src, 'r') as srcfile:
        for line in srcfile:
            links.append(line)
    links = [line.rstrip('\n') for line in links]
    return links


def process_input(test_image): 
    # convert image to numerical array and reshape dimensions to match neural net input
    x_input = kimage.img_to_array(test_image)
    x_input = np.expand_dims(x_input, axis=0)
    x_input = preprocess_input(x_input)
    return x_input


def display_predictions(pred, image=None, pause_after_show=True ):
    # aggregate data from predictions - parallel arrays for simplicity
    classes = []
    datapoints = []
    for cls in pred:
        classes.append( cls[1] )
        datapoints.append( cls[2] )
    item_index = np.arange(len(classes))
    
    if (image):
        # show image tested and corresponding machine predictions
        image_figure = plt.figure(1)
        plt.imshow(image)
        image_figure.show()

    prediction_figure = plt.figure(2)
    plt.xticks( item_index , classes)
    plt.ylabel('certainty')
    plt.bar(item_index, datapoints, align='center')
    prediction_figure.show()

    if pause_after_show is True:
        input()
 

# for every classname in the passed file path, pull image links
# and store them in the filesystem, return a table of labels for each file
def pull_classes(class_config_directory):
    dataset = {'id': [], 'label': []}

    # find class names to train on and attempt to download images for them
    training_classes = get_training_classes(class_config_directory)
    for cls in training_classes:
        if cls == '':
            continue
        print("Pulling images for class: {}".format(cls))
        path = './dataset/url/{}/url.txt'.format(cls)
        dest_path = './dataset/img/{}/'.format(cls)
        cls_links = read_links(path)
        imgs_pulled = download_imgs(cls_links, dest_path)
        for filename in imgs_pulled:
            dataset['id'].append(filename)
            dataset['label'].append(cls)
        print("\t{} images pulled: {}".format(cls, len(imgs_pulled)))

    # construct a table containing filenames and their corresponding classes
    df = pd.DataFrame(data=dataset)
    return df
     


# load a pre-existing, trained model, and add a new, blank classifier ontop
def newClassifier(target_size=(224,224), n_classes=14):
    base_model = DenseNet201(weights="imagenet")
    new_model = Sequential()
    
    base_model.layers.pop()
    #unwrap base_model layers and add them to a new Sequential model
    for layer in base_model.layers:
        layer.trainable = False
        new_model.add(layer)

    new_model.add( Dense(n_classes, activation="softmax") )
    return new_model



def main():
    # silence tensorflow's useless info logging
    logging.set_verbosity(logging.ERROR)

    # parse arguments (image file's path)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('classfile')
    argparser.add_argument('--skip-download', action='store_true')
    args = argparser.parse_args()

    dataset_frame = None
    if not (args.skip_download):
        # pull images and construct a table of image/label pairs
        dataset_frame = pull_classes(args.classfile)
        dataset_frame.to_csv('./dataset/dataset_cache')
    else:
        dataset_frame = pd.read_csv('./dataset/dataset_cache', index_col=0)


    dataset_frame['id'] = dataset_frame['id'].apply(lambda val:
        path.abspath(val)
    )

    # section off images for training and validation
    img_generator = kimage.ImageDataGenerator(validation_split=.25)
    data_flow = img_generator.flow_from_dataframe(
        directory=None,
        dataframe=dataset_frame,
        x_col="id",
        y_col="label",
        class_mode="categorical",
        target_size=(224, 224),
        subset="training"
    )
    validation_data_flow = img_generator.flow_from_dataframe(
        directory=None,
        dataframe=dataset_frame,
        x_col="id",
        y_col="label",
        class_mode="categorical",
        target_size=(224, 224),
        subset="validation"
    )

    # construct new classifier model from a pre-trained model
    new_model = newClassifier()
    new_model.summary()
    new_model.compile(optimizer=SGD(lr=.001, momentum=.9), loss="categorical_crossentropy", metrics=['accuracy'])
    
    new_model.fit_generator(
        data_flow,
        steps_per_epoch=10,
        epochs=5,
        validation_data=validation_data_flow,
        validation_steps=3
    )

    

    #pred = decode_predictions(base_model.predict(x_input), top=3)[0]
    #display_predictions(pred, test_image)
    return

main()

