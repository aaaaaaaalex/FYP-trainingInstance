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

from os import listdir, makedirs, path, environ
from io import BytesIO

from keras.applications.densenet import DenseNet201, preprocess_input
from keras.preprocessing import image as kimage
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from PIL import Image as pimage, ImageFile
from matplotlib import pyplot as plt
from tensorflow import logging as tflogging

import argparse
import requests
import h5py
import time
import numpy as np
import pandas as pd

class TrainingInstance():
    def __init__(self, 
            config_path = None,
            checkpoint_path = None,
            use_dataset_cache   = None,
            min_images_per_class = None):

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        if use_dataset_cache is None:
            self.use_dataset_cache = False
        else: self.use_dataset_cache = use_dataset_cache

        self.min_images_per_class = min_images_per_class

        return


    # wrap the Keras image data gen and manually normalize images
    @staticmethod
    def __flow_with_normalisation__(data_flow):
       for xs, ys in data_flow:
            np_xs = np.array(xs)
            norm_np_xs = np_xs / 255
            yield norm_np_xs, ys


    # attempts to validate if a collection of bytes represents an image by parsing them with PIL.Image
    # throws an exception if the data cannot be parsed
    @staticmethod
    def __sanitise_image__(bytes_data):
        buf = BytesIO(bytes_data)
        img = pimage.open(buf)

        # move data from original image into new image, dropping redundant metadata to save space / avoid warnings
        buf_no_exif = BytesIO()
        img_no_exif = pimage.new(img.mode, img.size)

        img_no_exif.putdata( list(img.getdata()) )
        img_no_exif.save(buf_no_exif, format="jpeg")

        # drop all EXIF data as it is all redundant
        return buf_no_exif.getvalue()


    #opens a file containing a newline-separated list of URLS, returns all URLS found
    @staticmethod
    def __read_links__(src):
        links = []
        with open(src, 'r') as srcfile:
            for line in srcfile:
                links.append(line)
        links = [line.rstrip('\n') for line in links]
        return links


    # ensure that there is the same number of data for every class in a dataframe
    def __unskew_data__ (self):

        # find the class with the smallest number of images
        lowest_images = -1
        lowest_classname = ""
        for classname in self.dataset_frame['label'].unique():
            class_count = len(self.dataset_frame[ self.dataset_frame['label'] == classname])
            if (lowest_images < 0 or class_count < lowest_images): 
                lowest_images = class_count
                lowest_classname = classname

        # drop images of other classes to un-skew the dataset
        all_classes = []
        print("\nClass '{}' has the least pulled-images. Image count: {}".format(lowest_classname, lowest_images))
        for classname in self.dataset_frame['label'].unique():
            all_imgs_for_class = self.dataset_frame[self.dataset_frame['label'] == classname]
            redundancy = len(all_imgs_for_class) - lowest_images
            lost_images = redundancy / len(all_imgs_for_class)

            if redundancy > 0 :
                all_imgs_for_class = all_imgs_for_class[:-redundancy]

            if self.min_images_per_class==None or len(all_imgs_for_class) >= self.min_images_per_class:
                all_classes.append(all_imgs_for_class)

            #print("{}: {} redundant images - {}% lost.".format(classname, redundancy, lost_images*100))
        
        self.dataset_frame = pd.concat(all_classes)
        return self.dataset_frame


    # downloads images at the URLs provided
    @staticmethod
    def __download_imgs__(links, save_dir):
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
                img = TrainingInstance.__sanitise_image__(r.content)
                filename = "{}img-{}.jpg".format(save_dir, i)
                if not path.exists(save_dir):
                    makedirs(save_dir)
                with open(filename, 'wb') as f:
                    f.write(img)
                images_pulled.append(filename)
            except (requests.exceptions.RequestException, OSError):
                pass
        
        return images_pulled


    # for every classname in the config path, pull image links
    # and store them in the filesystem, return a table of labels for each file
    def __pull_images__(self):
        dataset = {'id': [], 'label': []}

        # find class names to train on and attempt to download images for them
        training_classes = []
        with open(self.config_path) as fd:
            training_classes = [line.rstrip('\n') for line in fd]         

        # begin pulling images for each class
        for cls in training_classes:
            if cls == '':
                continue
            
            print("Pulling images for class: {}".format(cls))
            path      = './dataset/url/{}/url.txt'.format(cls)
            dest_path = './dataset/img/{}/'.format(cls)
            
            img_links = TrainingInstance.__read_links__(path)
            imgs_pulled = TrainingInstance.__download_imgs__(img_links, dest_path)
            for filename in imgs_pulled:
                dataset['id'].append(filename)
                dataset['label'].append(cls)
            print("\t{} images pulled: {}".format(cls, len(imgs_pulled)))

            # construct a table containing filenames and their corresponding classes
            df = pd.DataFrame(data=dataset)

            self.dataset_frame = df
            return self.dataset_frame


    # load a pre-existing, trained model, and add a new, blank classifier ontop
    def __make_classifier__(self):

        # construct new classifier model from a pre-trained model (or load a model from a checkpoint directory)
        if (self.checkpoint_path):
            filename = self.checkpoint_path + '/arch.json'
            f = open(filename, 'r')
            json = f.read()
            f.close()
            new_model = model_from_json(json)

            weights_filename = self.checkpoint_path + '/weights.h5'
            new_model.load_weights(weights_filename)
            print("Model loaded from checkpoint file")

        else:
            print("Model initialized from scratch.")
            base_model = DenseNet201(weights="imagenet", include_top=False)    

            #lock down the pre-trained layers for training
            for layer in base_model.layers:
                layer.trainable = False

            x = base_model.output
            
            x = Dense(128, activation="relu", name="top_dense")(x)
            x = Dropout(0.4)(x) # this aids in reducing overfitting
            x = GlobalAveragePooling2D(name="avg_pool")(x)

            output = Dense( len(self.classes), activation="softmax", name="classify")(x)
            new_model = Model(inputs=base_model.input, outputs=output)

        self.__model__ = new_model
        return self.__model__


    def init(self, unskew_data=True):
        if self.use_dataset_cache:
            self.dataset_frame = pd.read_csv('./dataset/dataset_cache', index_col=0)
        elif self.config_path is not None:
            self.__pull_images__()

        self.dataset_frame.to_csv('./dataset/dataset_cache')

        # Give absolute paths for images
        self.dataset_frame['id'] = self.dataset_frame['id'].apply( lambda val:
            path.abspath(val)
        )

        if unskew_data:
            self.__unskew_data__()

        self.classes = self.dataset_frame['label'].unique()
        self.__make_classifier__()
        return


    def train(
            self,
            epochs=15,
            training_batch_size=200,
            validation_batch_size=200,
            validation_split_ratio=0.2,
            lr=0.01, decay=0.0009):

        # shuffle data
        self.dataset_frame = self.dataset_frame.sample(frac=1).reset_index(drop=True)

        # section off images for training and validation
        # training params:
        validation_split_ratio   = .1
        total_training_samples   = int(len(self.dataset_frame) * (1.0-validation_split_ratio))
        total_validation_samples = int(len(self.dataset_frame) * validation_split_ratio)

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
            dataframe=self.dataset_frame,
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
            dataframe=self.dataset_frame,
            x_col="id",
            y_col="label",
            class_mode="categorical",
            target_size=(224, 224),
            subset="validation" ,
            batch_size=validation_batch_size
        )

        self.__model__.compile(optimizer=SGD(lr=0.01, decay=0.0009), loss="categorical_crossentropy", metrics=['categorical_accuracy'])

        # training callbacks
        estopper = EarlyStopping(monitor='val_categorical_accuracy', patience=2)

        print("\nTRAINING---------------------\nepochs: {},\nsteps_per_epoch: {},\nvalidation_steps: {}".format( epochs, steps_per_epoch, validation_steps))
        history = self.__model__.fit_generator(
            TrainingInstance.__flow_with_normalisation__(data_flow),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=TrainingInstance.__flow_with_normalisation__(validation_data_flow),
            validation_steps=validation_steps,
            callbacks=[estopper]
        )

        self.__model__ = self.__model__
        return (self.__model__, history)


    def save_model(self, output_directory="./out/{}{}"):
        model = self.__model__
        timestamp = int(time.time())

        # check if 'out' dir exists, create it if not
        if not path.exists(output_directory.format('','')):
            makedirs(output_directory.format('', ''))
        
        # repeat for unique, timestamped checkpoint directory
        if not path.exists( output_directory.format(timestamp, '') ):
            makedirs( output_directory.format(timestamp, '') )

        # write model architecture to JSON file
        with open(output_directory.format( timestamp, '/arch.json'), 'w') as f:
            f.write(model.to_json())
        
        # write weights to .h5 file
        model.save_weights(output_directory.format( timestamp, '/weights.h5'))

        save_dir = output_directory.format(timestamp, '')
        print("\nMODEL SAVED TO:{}".format(save_dir))
        return save_dir



def main():
    # silence tensorflow's useless info logging
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tflogging.set_verbosity(tflogging.ERROR)

    # parse arguments (image file's path)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config')
    argparser.add_argument('--checkpoint-dir')
    argparser.add_argument('--min-images-per-class', type=int)
    argparser.add_argument('--skip-download', action='store_true')
    args = argparser.parse_args()

    instance = TrainingInstance(
            config_path=args.config,
            checkpoint_path=args.checkpoint_dir,
            use_dataset_cache=args.skip_download,
            min_images_per_class=args.min_images_per_class
            )


    # start training
    instance.init()
    instance.train()
    instance.save_model()
    return

if __name__ == "__main__":
    main()
