from __future__ import print_function

import grpc

import forecast_pb2
import forecast_pb2_grpc


def main():

    channel = grpc.insecure_channel('localhost:50051')
    
    # Register Services to server
    stubForecast = forecast_pb2_grpc.ForecastAPIStub(channel)

    # Send and get response from Forecast Service
    # responseForecast = stubForecast.Forecast(forecast_pb2.ForecastRequest())
    # print(responseForecast)

    # Send and get response from TrainModel service
    trainRequest = forecast_pb2.TrainRequest(NumberEpochs = 1000)
    for trainResponse in stubForecast.RetrainModel(trainRequest):
        print(str(trainResponse))

if __name__ == '__main__':
    main()