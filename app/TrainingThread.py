import threading
import time

from os import listdir, makedirs, path
from DatasetCreator import DatasetCreator
from TrainingInstance import TrainingInstance


class TrainingThread(threading.Thread):

    def __init__(self, dbcursor, trainingargs,
            group=None, target=None, name=None):

        threading.Thread.__init__(self, group=group, target=target, name=name)

        self.trainingargs = trainingargs
        self.dbcursor = dbcursor
        self.instanceName = name
        return


    @staticmethod
    def save_model(self, model, output_directory="./instance/"):

        output_directory += "out/{}{}"

        # check if 'out' dir exists, create it if not
        if not path.exists(output_directory.format('','')):
            makedirs(output_directory.format('', ''))
        
        # repeat for unique, model_nameed checkpoint directory
        if not path.exists( output_directory.format(model_name, '') ):
            makedirs( output_directory.format(model_name, '') )

        # write model architecture to JSON file
        with open(output_directory.format( model_name, '/arch.json'), 'w') as f:
            f.write(model.to_json())
        
        # write weights to .h5 file
        model.save_weights(output_directory.format( model_name, '/weights.h5'))

        save_dir = output_directory.format(model_name, '')
        print("\nMODEL SAVED TO:{}".format(save_dir))
        return save_dir


    def run(self):
        if not path.exists('./instances/'):
            makedirs('./instances/')

        workdir = './instances/{}/'.format(self.instanceName)
        if not path.exists(workdir):
            makedirs(workdir)

        dataset_creator = DatasetCreator(
                self.dbcursor,
                workdir=workdir,
                class_config=self.trainingargs['class_config'])
        dataset_dataframe = dataset_creator.get_dataset_dataframe()

        instance = TrainingInstance(
                dataset_dataframe=dataset_dataframe)
        
        # start training
        model = instance.train()
        instance.save_model(model, workdir)

        return
