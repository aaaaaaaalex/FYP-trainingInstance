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
    def save_model(model, output_directory):

        output_directory += "out/{}"
        arch_path = output_directory.format('arch.json')
        weights_path = output_directory.format('weights.h5')

        # check if 'out' dir exists, create it if not
        if not path.exists(output_directory.format('')):
            makedirs(output_directory.format(''))
        
        # repeat for unique, model_nameed checkpoint directory
        if not path.exists( output_directory.format('') ):
            makedirs( output_directory.format('') )

        # write model architecture to JSON file
        with open(arch_path, 'w') as f:
            f.write(model.to_json())
        
        # write weights to .h5 file
        model.save_weights(weights_path)
        
        return (arch_path, weights_path) 


    def run(self):
        if not path.exists('./instances/'):
            makedirs('./instances/')

        workdir = './instances/{}/'.format(self.instanceName)
        if not path.exists(workdir):
            makedirs(workdir)

        # TODO: sanitize classlist - currently, erronious (possibly dangerous) classnames will reach the SQL request untouched

        dataset_creator = DatasetCreator(
                self.dbcursor,
                workdir=workdir,
                class_config=self.trainingargs['class_config'])

        dataset_dataframe = dataset_creator.get_dataset_dataframe()
        checkpoint_path = workdir+'out/' if path.exists(workdir+'out') else None

        instance = TrainingInstance(
                dataset_dataframe=dataset_dataframe,
                checkpoint_path=checkpoint_path)
        
        # start training
        model, _ = instance.train()
        TrainingThread.save_model(model, output_directory=workdir)

        logging.info('Training Complete for model {}'.format(self.instanceName))
        exit()
