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

import argparse
import requests
import h5py
import time
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


class TrainingInstance():
    def __init__(self):
        return

    # wrap the Keras image data gen and manually normalize images
    @staticmethod
    def flow_with_normalisation(data_flow):
       for xs, ys in data_flow:
            np_xs = np.array(xs)
            norm_np_xs = np_xs / 255
            yield norm_np_xs, ys


    # attempts to validate if a collection of bytes represents an image by parsing them with PIL.Image
    # throws an exception if the data cannot be parsed
    def sanitise_image(self, bytes_data):
        buf = BytesIO(bytes_data)
        img = pimage.open(buf)

        # move data from original image into new image, dropping redundant metadata to save space / avoid warnings
        buf_no_exif = BytesIO()
        img_no_exif = pimage.new(img.mode, img.size)

        img_no_exif.putdata( list(img.getdata()) )
        img_no_exif.save(buf_no_exif, format="jpeg")

        # drop all EXIF data as it is all redundant
        return buf_no_exif.getvalue()

    def read_links(self, src):
        links = []
        with open(src, 'r') as srcfile:
            for line in srcfile:
                links.append(line)
        links = [line.rstrip('\n') for line in links]
        return links

    def get_training_classes(self, classfile='./dataset/classes.config'):
        classes = []
        with open(classfile) as fd:
            classes = [line.rstrip('\n') for line in fd]         
        return classes

    def download_imgs(self, links, save_dir, download_limit=1000):
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
                img = self.sanitise_image(r.content)
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
    def pull_dataset(self, class_config_directory, min_cutoff=None):
        dataset = {'id': [], 'label': []}

        # find class names to train on and attempt to download images for them
        training_classes = self.get_training_classes(class_config_directory)
        for cls in training_classes:
            if cls == '':
                continue
            print("Pulling images for class: {}".format(cls))
            path = './dataset/url/{}/url.txt'.format(cls)
            dest_path = './dataset/img/{}/'.format(cls)
            cls_links = self.read_links(path)
            imgs_pulled = self.download_imgs(cls_links, dest_path)
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
    def newClassifier(self, n_classes):
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


    def train(
            self,
            dataset_frame,
            checkpoint=None,
            epochs=15,
            training_batch_size=256,
            validation_batch_size=256,
            validation_split_ratio=0.2,
            lr=0.01, decay=0.0009):

        # Give absolute paths for images, then shuffle the dataset
        dataset_frame['id'] = dataset_frame['id'].apply(lambda val:
            path.abspath(val)
        )
        dataset_frame = dataset_frame.sample(frac=1).reset_index(drop=True)

        # section off images for training and validation
        # training params:
        validation_split_ratio   = .1
        total_training_samples   = int(len(dataset_frame) * (1.0-validation_split_ratio))
        total_validation_samples = int(len(dataset_frame) * validation_split_ratio)

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

        # construct new classifier model from a pre-trained model (or load a model from a checkpoint directory)
        print("\nMODEL---------------------")
        classes = dataset_frame['label'].unique()
        new_model = None
        if (checkpoint):
            filename = checkpoint+'/arch.json'
            f = open(filename, 'r')
            json = f.read()
            f.close()
            new_model = model_from_json(json)

            weights_filename = checkpoint+'/weights.h5'
            new_model.load_weights(weights_filename)

            print("Model loaded from checkpoint file")
        else:
            new_model = self.newClassifier(n_classes=len(classes))
            print("Model initialized from scratch.")
        
        new_model.compile(optimizer=SGD(lr=0.01, decay=0.0009), loss="categorical_crossentropy", metrics=['categorical_accuracy'])

        # training callbacks
        estopper = EarlyStopping(monitor='val_categorical_accuracy', patience=2)

        print("\nTRAINING---------------------\nepochs: {},\nsteps_per_epoch: {},\nvalidation_steps: {}".format( epochs, steps_per_epoch, validation_steps))
        history = new_model.fit_generator(
            TrainingInstance.flow_with_normalisation(data_flow),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=flow_with_normalisation(validation_data_flow),
            validation_steps=validation_steps,
            callbacks=[estopper]
        )

        self.__model__ = new_model
        return (new_model, history)

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
        print("\nMODEL SAVED TO:{}")
        return output_directory.format(timestamp, '')


def main():
    # silence tensorflow's useless info logging
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # parse arguments (image file's path)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config')
    argparser.add_argument('--checkpoint-dir')
    argparser.add_argument('--min-images-per-class', type=int)
    argparser.add_argument('--skip-download', action='store_true')
    args = argparser.parse_args()

    instance = TrainingInstance()

    print("\nDATASET---------------------")
    if args.config:
        # pull images and construct a table of image/label pairs
        dataset_frame = instance.pull_dataset(args.config, args.min_images_per_class)
        dataset_frame.to_csv('./dataset/dataset_cache')
    elif args.skip_download:
        dataset_frame = pd.read_csv('./dataset/dataset_cache', index_col=0)
    else:
        print("Invalid args: no config for pulling images was provided, and --skip-download was not declared")
        return

    # start training
    if (args.checkpoint_dir):
        (model, _) = instance.train(dataset_frame, checkpoint=args.checkpoint_dir)
    else:
        (model, _) = instance.train(dataset_frame)

    instance.save_model(model)

    #pred = decode_predictions(base_model.predict(x_input), top=3)[0]
    #display_predictions(pred, test_image)
    return

if __name__ == "__main__":
    main()
