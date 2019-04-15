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

from os import makedirs, path, environ

from keras.applications.densenet import DenseNet201, preprocess_input
from keras.preprocessing import image as kimage
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from matplotlib import pyplot as plt
from tensorflow import logging as tflogging

import argparse
import h5py
import time
import numpy as np

class TrainingInstance():
    def __init__(self,
            dataset_dataframe,
            checkpoint_path = None):

        self.dataset_dataframe = dataset_dataframe
        self.classes = self.dataset_dataframe['label'].unique()
        self.checkpoint_path = checkpoint_path

        self.__make_classifier__()
        return


    # wrap the Keras image data gen and manually normalize images
    @staticmethod
    def __flow_with_normalisation__(data_flow):
       for xs, ys in data_flow:
            np_xs = np.array(xs)
            norm_np_xs = np_xs / 255
            yield norm_np_xs, ys


    # load a pre-existing, trained model, and add a new, blank classifier ontop
    def __make_classifier__(self):
        # construct new classifier model from a pre-trained model (or load a model from a checkpoint directory)
        if (self.checkpoint_path):
            arch_filename = self.checkpoint_path + 'arch.json'
            weights_filename = self.checkpoint_path + 'weights.h5'

            f = open(arch_filename, 'r')
            json = f.read()
            f.close()
            
            new_model = model_from_json(json)
            new_model.load_weights(weights_filename)

            print("Model loaded from checkpoint file")

        else:
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


    def train(
            self,
            epochs=15,
            training_batch_size=50,
            validation_batch_size=50,
            validation_split_ratio=0.2,
            lr=0.01, decay=0.0009):

        # section off images for training and validation
        # training params:
        validation_split_ratio   = .1
        total_training_samples   = int(len(self.dataset_dataframe) * (1.0-validation_split_ratio))
        total_validation_samples = int(len(self.dataset_dataframe) * validation_split_ratio)

        steps_per_epoch = int(total_training_samples / training_batch_size)
        validation_steps = int(total_validation_samples / validation_batch_size)

        if validation_steps < 1: raise ValueError('Validation steps for training was below 1, this dataset is likely too small!')

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
            dataframe=self.dataset_dataframe,
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
            dataframe=self.dataset_dataframe,
            x_col="id",
            y_col="label",
            class_mode="categorical",
            target_size=(224, 224),
            subset="validation" ,
            batch_size=validation_batch_size
        )

        model = self.__model__
        model.compile(optimizer=SGD(lr=0.01, decay=0.0009), loss="categorical_crossentropy", metrics=['categorical_accuracy'])

        # training callbacks
        estopper = EarlyStopping(monitor='val_categorical_accuracy', patience=2)

        print("\nTRAINING---------------------\nepochs: {},\nsteps_per_epoch: {},\nvalidation_steps: {}".format( epochs, steps_per_epoch, validation_steps))
        history = model.fit_generator(
            TrainingInstance.__flow_with_normalisation__(data_flow),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=TrainingInstance.__flow_with_normalisation__(validation_data_flow),
            validation_steps=validation_steps,
            callbacks=[estopper]
        )

        self.__model__ = model
        return (model, history)



def main():
    # silence tensorflow's useless info logging
    #environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #tflogging.set_verbosity(tflogging.ERROR)

    # parse arguments (image file's path)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--checkpoint-dir')
    argparser.add_argument('--class-config')
    argparser.add_argument('--unskew-data',       action='store_true')
    argparser.add_argument('--shuffle-data',      action="store_true")
    args = argparser.parse_args()


    # shuffle data
    dataset_dataframe = dataset_dataframe.sample(frac=1).reset_index(drop=True)
    
    instance = TrainingInstance(
            dataset_dataframe=dataset_dataframe,
            checkpoint_path=args.checkpoint_dir,
            )

    # start training
    instance.train()
    instance.save_model()
    return

if __name__ == "__main__":
    main()
