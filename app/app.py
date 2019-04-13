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
                self.dbcursor = self.db.cursor(buffered=True)
                print("DB connection established!")
                break
            except connector.errors.InterfaceError:
                sleep(5)
                continue
        return


    def TrainModel (self, train_request, cont):
        logging.info("TrainModel request: {}".format(train_request))

        # parse args
        trainingargs = {
                'class_config' : json.loads(train_request.classlist),
            }
        thr = TrainingThread(self.dbcursor, trainingargs, name=int(time()) )
        thr.start()

        response = TrainResponse(response="Training Started")
        return response


    def GetActiveTrainingInstances (self, req, cont):
        logging.info("GetActiveTrainingInstances request: {}".format(req))

        return TrainingInstanceList(
                    instances=[TrainingInstanceInfo( date_started="666th" , classlist="[ducks, haha's, the beegees]", classlist_locations="[haha nothing here boiii]" )])



_ONE_DAY_IN_SECONDS = (60*60) * 24
def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port('[::]:8081')
    serve_training_servicer(TrainingServicer(), server)

    server.start()
    try:
        while True:
            sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        print("bye!")
    server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
