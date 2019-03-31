import grpc
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import trainingInstance
from TrainingService_pb2 import TrainingInstance, TrainingInstanceList, TrainRequest, TrainResponse, InstanceFilter
from TrainingService_pb2_grpc import TrainingServiceServicer, add_TrainingServiceServicer_to_server as serve_training_servicer
import TrainingService_pb2_grpc


class TrainingServicer (TrainingServiceServicer):
    def __init__(self):
        return

    def TrainModel (self, req, cont):
        print("Called!")
        print("req: {}, \n\n cont: {}".format(req, cont))
        
        response = TrainResponse()
        return response

    def GetActiveTrainingInstances (self, req, cont):
        print("Called!")
        print("req: {}, \n\n cont: {}".format(req, cont))

        return TrainingInstanceList(
                    [TrainingInstance( date_started="420th" , classlist="[ducks, janMichaelVincents, deathGrips]", classlist_locations="[haha nothing here boiii]" )])


_ONE_DAY_IN_SECONDS = (60*60) * 24

def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port('[::]:8081')
    serve_training_servicer(TrainingServicer(), server)

    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
