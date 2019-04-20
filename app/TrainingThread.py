import logging
import threading
import time
import datetime

from DatasetCreator import DatasetCreator
from keras import backend as K
from os import listdir, makedirs, path
from shutil import rmtree
from TrainingInstance import TrainingInstance

class TrainingThread(threading.Thread):

    def __init__(self, db, trainingargs,
            group=None, target=None, name=None):

        threading.Thread.__init__(self, group=group, target=target, name=name)

        self.trainingargs = trainingargs
        self.db = db
        self.instanceName = name

        K.clear_session()
        return


    @staticmethod
    def save_model(model, output_directory):
        output_directory += "out/{}"
        dir_path = output_directory.format('')
        arch_path = output_directory.format('arch.json')
        weights_path = output_directory.format('weights.h5')

        if path.exists(dir_path):
            rmtree(dir_path)
        makedirs(dir_path)

        # write model architecture to JSON file
        with open(arch_path, 'w') as f:
            f.write(model.to_json())
        
        # write weights to .h5 file
        model.save_weights(weights_path)
        
        return (arch_path, weights_path)


    def getModelClasses(self, modelName):
        cursor = self.db.cursor()
        cursor.execut("""
            SELECT cName from Category
            WHERE cID IN (
                SELECT cID from ModelCategory
                WHERE modelName = {});
            """.format(modelName))
        
        classnames = cursor.fetchall()
        return


    def insertModel(self, history, modelname):
        cursor = self.db.cursor()
        modelCAccuracy = float(history.history['val_categorical_accuracy'][-1])
        modelDateCreated = datetime.datetime.now()
        modelURL = None
        modelName = str(self.instanceName)

        cursor.execute("""
            SELECT count(modelName)
            FROM Model
            WHERE modelName={}
            """.format(modelName))

        cnt = cursor.fetchone()[0]

        if int(cnt) > 0:
            cursor.execute("""
                UPDATE Model
                SET
                    modelCAccuracy=%s,
                    modelDateCreated=%s
                WHERE modelName=%s;
                """, (modelCAccuracy, modelDateCreated, modelName))

        else:
            cursor.execute("""
                INSERT INTO `Model`
                (modelCAccuracy, modelDateCreated, modelURL, modelName)
                VALUES (%s, %s, %s, %s);
                """, (modelCAccuracy, modelDateCreated, modelURL, modelName))

            # cosntruct comma-separated list string for FIND_IN_SET sql function
            classlist_str = ""
            for classname in self.trainingargs['class_config']:
                classlist_str += "{},".format(classname)

            # get ID's for the model's categories
            cursor.execute("""
                SELECT cID FROM Category
                WHERE FIND_IN_SET(cName, '{}');
            """.format(classlist_str))

            class_ids = cursor.fetchall()

            for cl in [tpl[0] for tpl in class_ids]:
                cursor.execute(""" 
                    INSERT INTO ModelCategory
                    ( mcID, cID, modelName )
                    VALUES (%s, %s, %s);
                """, (None, cl, modelName) )

        self.db.commit()
        return


    def run(self):
        self._is_running = True

        # TODO: sanitize classlist - currently, erronious (possibly dangerous) classnames will reach the SQL request untouched
        if not path.exists('./instances/'):
            makedirs('./instances/')

        workdir = './instances/{}/'.format(self.instanceName)
        if not path.exists(workdir):
            makedirs(workdir)

        dataset_creator = DatasetCreator(
                self.db.cursor(),
                workdir=workdir,
                class_config=self.trainingargs['class_config'])
        dataset_dataframe = dataset_creator.get_dataset_dataframe()
        self.trainingargs['class_config'] = dataset_dataframe['label'].unique()

        checkpointarg = self.trainingargs['checkpoint_name']
        checkpointarg = './instances/{}/out/'.format(checkpointarg) if checkpointarg else None
        if checkpointarg and not path.exists(checkpointarg): checkpointarg = None

        instance = TrainingInstance(
                dataset_dataframe=dataset_dataframe,
                checkpoint_path=checkpointarg)
        
        # start training
        model, history = instance.train()
        self.insertModel(history, self.instanceName)
        TrainingThread.save_model(model, output_directory=workdir)

        logging.info('Training Complete for model {}'.format(self.instanceName))

        self._is_running = False
        exit()
