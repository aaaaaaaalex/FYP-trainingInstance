import grpc
import json
import logging
from time import sleep, time
from mysql import connector
from TrainingThread import TrainingThread
from concurrent.futures import ThreadPoolExecutor
from TrainingService_pb2 import TrainingInstanceInfo, TrainingInstanceList, TrainRequest, TrainResponse 
from TrainingService_pb2_grpc import TrainingServiceServicer, add_TrainingServiceServicer_to_server as serve_training_servicer


class TrainingServicer (TrainingServiceServicer):
    def __init__(self):
        while True:
            try:
                self.db = connector.connect(
                    host="db",
                    user="root",
                    passwd="test",
                    database='app'
                )
                logging.info("DB connection established!")
                break
            except connector.errors.InterfaceError:
                sleep(5)
                continue

        self.training_threads = []
        return


    def TrainModel (self, train_request, cont):
        logging.info("TrainModel request: {}".format(train_request))

        if (len(self.training_threads)>0 and self.training_threads[0]._is_running):
            return TrainResponse(status=409, response="Cannot start training: Max number of training threads is reached. To train a new instance, either cancel an active instance, or wait for one to finish.")

        # check args
        if ((train_request.classlist and len(train_request.classlist) > 1)
                or (train_request.instance_name and len(train_request.instance_name) > 1)):
            # parse args
            trainingName = train_request.instance_name if train_request.instance_name else str(int(time()))
            checkpoint = train_request.checkpoint_name if train_request.checkpoint_name else None
            trainingargs = {
                    'class_config' : json.loads(train_request.classlist) if train_request.classlist else None,
                    'checkpoint_name' : checkpoint
                }
                
            thr = TrainingThread(self.db, trainingargs, name=trainingName)
            thr.start()
            self.training_threads.append(thr)

            response = TrainResponse(status=200, response="Training Started.", instance_name=trainingName)
        else: response = TrainResponse(status=422, response="No classlist or checkpoint_name specified.")
        return response


    def GetActiveTrainingInstances (self, req, cont):
        logging.info("GetActiveTrainingInstances request: {}".format(req))
        return TrainingInstanceList(
                    instances=[TrainingInstanceInfo( date_started="666th" , classlist="[ducks, haha's, the beegees]", classlist_locations="[haha nothing here boiii]" )])



_ONE_DAY_IN_SECONDS = (60*60) * 24
def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port('[::]:8081')
    trainingServicer = TrainingServicer()
    serve_training_servicer(trainingServicer, server)

    server.start()
    try:
        while True:
            sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        print("bye!")

    # wait for dangling threads
    for thread in trainingServicer.trainingThreads:
        thread.join()

    server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
