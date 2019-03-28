import grpc
import time
import logging
from concurrent import futures

import trainingInstance


_ONE_DAY_IN_SECONDS = (60*60) * 24




def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port('[::]:8081')
    #helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)

    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
