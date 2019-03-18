# ----------------------------------------------------------------------------------------------|
# -------------------------------------------- DISCLAIMER FOR THE USE OF DENSENET WEIGHTS ------|
# ----------------------------------------------------------------------------------------------|
# Copyright (c) 2016, Zhuang Liu.                                                               |
# All rights reserved.                                                                          |
#                                                                                               |
# Redistribution and use in source and binary forms, with or without modification,              |
# are permitted provided that the following conditions are met:                                 |
#                                                                                               |
#  * Redistributions of source code must retain the above copyright notice, this                |
#    list of conditions and the following disclaimer.                                           |
#                                                                                               |
#  * Redistributions in binary form must reproduce the above copyright notice,                  |
#    this list of conditions and the following disclaimer in the documentation                  |
#    and/or other materials provided with the distribution.                                     |
#                                                                                               |
#  * Neither the name DenseNet nor the names of its contributors may be used to                 |
#    endorse or promote products derived from this software without specific                    |
#    prior written permission.                                                                  |
#                                                                                               |
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND               |
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                 |
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                        |
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR              |
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                |
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                  |
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON                |
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                       |
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                 |
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                  |
# ----------------------------------------------------------------------------------------------|

from os import listdir, makedirs, path
from io import BytesIO

from keras.applications.densenet import DenseNet201, preprocess_input
from keras.preprocessing import image as kimage
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from tensorflow import logging
from PIL import Image as pimage, ImageFile
from matplotlib import pyplot as plt

import argparse
import requests
import numpy as np
import pandas as pd

# def process_input(test_image): 
#     # convert image to numerical array and reshape dimensions to match neural net input
#     x_input = kimage.img_to_array(test_image)
#     x_input = np.expand_dims(x_input, axis=0)
#     x_input = preprocess_input(x_input)
#     return x_input


# def display_predictions(pred, image=None, pause_after_show=True ):
#     # aggregate data from predictions - parallel arrays for simplicity
#     classes = []
#     datapoints = []
#     for cls in pred:
#         classes.append( cls[1] )
#         datapoints.append( cls[2] )
#     item_index = np.arange(len(classes))
    
#     if (image):
#         # show image tested and corresponding machine predictions
#         image_figure = plt.figure(1)
#         plt.imshow(image)
#         image_figure.show()

#     prediction_figure = plt.figure(2)
#     plt.xticks( item_index , classes)
#     plt.ylabel('certainty')
#     plt.bar(item_index, datapoints, align='center')
#     prediction_figure.show()

#     if pause_after_show is True:
#         input()




# attempts to validate if a collection of bytes represents an image by parsing them with PIL.Image
# throws an exception if the data cannot be parsed
def sanitise_image(bytes_data):
    buf = BytesIO(bytes_data)
    img = pimage.open(buf)

    # move data from original image into new image, dropping redundant metadata to save space / avoid warnings
    buf_no_exif = BytesIO()
    img_no_exif = pimage.new(img.mode, img.size)

    img_no_exif.putdata( list(img.getdata()) )
    img_no_exif.save(buf_no_exif, format="jpeg")

    # drop all EXIF data as it is all redundant
    return buf_no_exif.getvalue()

def read_links(src):
    links = []
    with open(src, 'r') as srcfile:
        for line in srcfile:
            links.append(line)
    links = [line.rstrip('\n') for line in links]
    return links

def get_training_classes(classfile='./dataset/classes.config'):
    classes = []
    with open(classfile) as fd:
        classes = [line.rstrip('\n') for line in fd]         
    return classes

def download_imgs(links, save_dir, download_limit=1000):
    # throw error if links isnt a list
    assert type(links) is list
    images_pulled = []
    
    # iterate over each link, carrying both the link and it's list index
    i=0
    for link, i in zip(links, range(len(links))):
        try:
            # make a GET request and dont follow redirects - timeout after 3 secs
            r = requests.get(link, allow_redirects=False, timeout=3)
            r.raise_for_status()

            # make sure the response is an image, not HTML
            img = sanitise_image(r.content)
            filename = "{}img-{}.jpg".format(save_dir, i)
            if not path.exists(save_dir):
                makedirs(save_dir)
            with open(filename, 'wb') as f:
                f.write(img)
            images_pulled.append(filename)
        except (requests.exceptions.RequestException, OSError):
            pass
    
    return images_pulled

# for every classname in the passed file path, pull image links
# and store them in the filesystem, return a table of labels for each file
def pull_dataset(class_config_directory, min_cutoff=None):
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

    # find the class with the smallest number of images
    # print(df)
    lowest_images = -1
    lowest_classname = ""
    for classname in df['label'].unique():
        class_count = len(df[ df['label'] == classname])
        if (lowest_images < 0 or class_count < lowest_images): 
            lowest_images = class_count
            lowest_classname = classname

    # drop images of other classes to un-skew the dataset
    all_classes = []
    print("\nClass '{}' has the least pulled-images. Image count: {}".format(lowest_classname, lowest_images))
    for classname in df['label'].unique():
        all_imgs_for_class = df[df['label'] == classname]
        redundancy = len(all_imgs_for_class) - lowest_images
        lost_images = redundancy / len(all_imgs_for_class)
        if redundancy > 0 :
            all_imgs_for_class = all_imgs_for_class[:-redundancy]
        if min_cutoff==None or len(all_imgs_for_class) >= min_cutoff:
            all_classes.append(all_imgs_for_class)
        print("{}: {} redundant images - {}% lost.".format(classname, redundancy, lost_images*100))
    
    df = pd.concat(all_classes)
    print(df)

    return df


# load a pre-existing, trained model, and add a new, blank classifier ontop
def newClassifier(n_classes):
    base_model = DenseNet201(weights="imagenet", include_top=False)    

    #lock down the pre-trained layers for training
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    
    x = Dense(128, activation="relu", name="top_dense")(x)
    x = Dropout(0.4)(x) # this aids in reducing overfitting
    x = GlobalAveragePooling2D(name="avg_pool")(x)

    output = Dense(n_classes, activation="softmax", name="classify")(x)
    model = Model(inputs=base_model.input, outputs=output)

    return model


# wrap the Keras image data gen and manually normalize images
def flow_with_normalisation(data_flow):
    for xs, ys in data_flow:
        np_xs = np.array(xs)
        norm_np_xs = np_xs / 255
        yield norm_np_xs, ys


def main():
    # silence tensorflow's useless info logging
    logging.set_verbosity(logging.ERROR)

    # parse arguments (image file's path)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('classfile')
    argparser.add_argument('--min-images-per-class')
    argparser.add_argument('--skip-download', action='store_true')
    args = argparser.parse_args()

    dataset_frame = None
    if not (args.skip_download):
        # pull images and construct a table of image/label pairs
        dataset_frame = pull_dataset(args.classfile, args.min_images_per_class)
        dataset_frame.to_csv('./dataset/dataset_cache')
    else:
        dataset_frame = pd.read_csv('./dataset/dataset_cache', index_col=0)

    # Give absolute paths for images, then shuffle the dataset
    dataset_frame['id'] = dataset_frame['id'].apply(lambda val:
        path.abspath(val)
    )
    dataset_frame = dataset_frame.sample(frac=1).reset_index(drop=True)



    # section off images for training and validation
    # training params:
    validation_split_ratio   = .2
    total_training_samples   = int(len(dataset_frame) * (1.0-validation_split_ratio))
    total_validation_samples = int(len(dataset_frame) * validation_split_ratio)

    epochs = 10
    training_batch_size = 256
    validation_batch_size = 256

    steps_per_epoch = int(total_training_samples / training_batch_size)
    validation_steps = int(total_validation_samples / validation_batch_size)

    img_generator = kimage.ImageDataGenerator(
        validation_split=validation_split_ratio,
        #rescale=1.0/255.0,
        #vertical_flip=True,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
    )
    data_flow = img_generator.flow_from_dataframe(
        shuffle=True,
        directory=None,
        dataframe=dataset_frame,
        x_col="id",
        y_col="label",
        class_mode="categorical",
        target_size=(224, 224),
        subset="training",
        batch_size=training_batch_size
    )
    validation_data_flow = img_generator.flow_from_dataframe(
        shuffle=False,
        directory=None,
        dataframe=dataset_frame,
        x_col="id",
        y_col="label",
        class_mode="categorical",
        target_size=(224, 224),
        subset="validation" ,
        batch_size=validation_batch_size
    )

    # for (x, y) in validation_data_flow:
    #     x_np = np.array(x[0])
    #     print(x_np.shape)
    #     print("X----------------------------------------------\n{}\nY----------------------------------------------{}\n\n\n".format(x[0], y))
    #     plt.imshow(x_np/255)
    #     plt.pause(.5)
    
    # return

    # construct new classifier model from a pre-trained model
    classes = dataset_frame['label'].unique()
    new_model = newClassifier(n_classes=len(classes))

    new_model.compile(optimizer=SGD(lr=0.01, decay=0.0009), loss="categorical_crossentropy", metrics=['categorical_accuracy'])

    print("number of training samples: {},\nnumber of validation samples: {},\nepochs: {},\nsteps_per_epoch: {},\nvalidation_steps: {}".format(total_training_samples, total_validation_samples, epochs, steps_per_epoch, validation_steps))
    new_model.fit_generator(
        flow_with_normalisation(data_flow),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=flow_with_normalisation(validation_data_flow),
        validation_steps=validation_steps
    )

    #pred = decode_predictions(base_model.predict(x_input), top=3)[0]
    #display_predictions(pred, test_image)
    return

main()

