import math
import joblib
import datetime
import warnings
import os
import atexit
import time
import threading

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import Callback

import grpc
from grpc_status import rpc_status

from google.protobuf import any_pb2
from google.rpc import code_pb2, status_pb2, error_details_pb2

import forecast_pb2
import forecast_pb2_grpc

from concurrent import futures
from observable import Observable

observer = Observable()
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

# Define Initial Training Progress
PROGRESS = 0.0

# Track Training Progress
@observer.on('progress')
def changeProgress(value):
    global PROGRESS
    PROGRESS = value

class ForecastModel():

    def __init__(self, attributes=["Revenue","Shipment","Order"], n_steps_in=120, n_steps_out=90, n_epochs=800, n_neurons=300, datasetPath="./Data Science_20200214.xlsx", scalerPath="./default_scaler.pkl", modelPath="./default_model.h5"):
        # Attributes
        self.attributeSelected = attributes
        
        # File Path
        self.datasetPath = datasetPath
        self.scalerPath = scalerPath
        self.modelPath = modelPath

        # Model Parameters
        self.n_steps_in, self.n_steps_out = n_steps_in, n_steps_out
        self.n_epochs, self.n_neurons = n_epochs, n_neurons
        self.n_features = len(self.attributeSelected)
        
    def SplitSequence(self, sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def LoadDataset(self):
        # read dataset from file
        if(".xlsx" in self.datasetPath or ".xls" in self.datasetPath):
            self.df = pd.read_excel(self.datasetPath, index_col="Date", parse_dates=True)
        elif(".csv" in self.datasetPath):
            self.df = pd.read_csv(self.datasetPath, index_col="Date", parse_dates=True)

        self.df = self.df.sort_index(ascending=True)
        self.df = self.df[self.attributeSelected]
    
    def TransformDataset(self):
        # normalize data into 0 to 1 using MinMaxScaler
        self.scaler = MinMaxScaler()

        self.df_transformed = self.scaler.fit_transform(self.df)

        # Save Scaler
        # if(os.path.exists(self.scalerPath)):
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.scalerPath = "./service_dist/scaler_" + str(self.n_steps_in) + "_" + str(self.n_steps_out) + "_" + str(self.n_features) + "_" + now + ".pkl"

        joblib.dump(self.scaler, self.scalerPath)

        # split into samples
        self.trainX, self.trainY = self.SplitSequence(self.df_transformed, self.n_steps_in, self.n_steps_out)

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        self.trainX = self.trainX.reshape((self.trainX.shape[0], self.n_steps_in, self.n_features))
        self.trainY = self.trainY.reshape((self.trainY.shape[0], self.n_steps_out, self.n_features))

    def TrainModel(self):
        # Define Model
        self.model = Sequential()
        self.model.add((LSTM(self.n_neurons, activation='tanh', input_shape=(self.n_steps_in, self.n_features))))
        self.model.add(RepeatVector(self.n_steps_out))
        self.model.add((LSTM(self.n_neurons, activation='tanh', return_sequences=True)))
        self.model.add(TimeDistributed(Dense(self.n_features)))
        self.model.compile(optimizer='adam', loss='mse')
        
        # Fit Model
        self.model.fit(self.trainX[:-1], self.trainY[:-1], epochs=self.n_epochs, verbose=1, callbacks=[TrainingProgressCallback()])

        # Save Model
        # if(os.path.exists(self.modelPath)):
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.modelPath = "./service_dist/model_" + str(self.n_steps_in) + "_" + str(self.n_steps_out) + "_" + str(self.n_features) + "_" + now + ".h5"
        self.model.save(self.modelPath)

    def ValidateModel(self):
        # Validate Model
        validate_X = self.trainX[-1].reshape(((1, self.n_steps_in, self.n_features)))
        validate_Y = self.model.predict(validate_X)
        validate_Y = self.scaler.inverse_transform(validate_Y[0])

        attributes_Validate = [attribute + "_Pred" for attribute in self.attributeSelected]

        df_validate = self.df.copy()
        df_validate = df_validate[-(self.n_steps_out):]
        df_validate = pd.concat([df_validate, pd.DataFrame(validate_Y, columns=attributes_Validate, index=df_validate.index)], axis=1)
        df_validate.index.name = "Date"
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.validateDataFilename = "./service_dist/model_validation_data" + now + ".csv"
        df_validate.to_csv(self.validateDataFilename)

        validationDataFile = open(self.validateDataFilename, mode='r')
        self.validateData = validationDataFile.read()
        validationDataFile.close()

        self.validateRMSEScore = []
        self.validateFinalValueError = []

        # Get Validation RMSE and Final Max Value Error (%)
        for n in range(int(df_validate.columns.size/2)):
            # Get RMSE of Actual and Predicted values
            self.validateRMSEScore.append(math.sqrt(mean_squared_error(df_validate[df_validate.columns[n]], df_validate[df_validate.columns[n+(df_validate.columns.size/2)]])))
            print('Validate Score: %.2f RMSE for %s' % (self.validateRMSEScore[n], df_validate.columns[n]))

            # Get Largest Nth Values from DataFrame for Checking on Next Final Value Prediction
            df_validate_nLargest = df_validate.nlargest(10,df_validate.columns[n])
            df_validate_pred_nLargest = df_validate.nlargest(10,df_validate.columns[n+(df_validate.columns.size/2)])

            timediff = np.diff(df_validate_nLargest.index.values.astype(dtype='datetime64[D]')).astype('int')
            timediff_pred = np.diff(df_validate_pred_nLargest.index.values.astype(dtype='datetime64[D]')).astype('int')

            index = 0
            index_pred = 0
            
            if(any(timediff > 30)):
                
                timediff = timediff[:(np.argmax(timediff > 30) + 1)]

                if(any(timediff < -30)):
                    index = np.argmax(timediff < 30)
                else:
                    index = np.argmax(timediff > 30) + 1

            if(any(timediff_pred > 30)):
                
                timediff = timediff_pred[:(np.argmax(timediff_pred > 30) + 1)]

                if(any(timediff_pred < -30)):
                    index_pred = np.argmax(timediff_pred < 30)
                else:
                    index_pred = np.argmax(timediff_pred > 30) + 1

            maxValueActual = df_validate[df_validate.columns[n]].loc[df_validate_nLargest.index[index]]
            maxValuePred = df_validate[df_validate.columns[n+(df_validate.columns.size/2)]].loc[df_validate_pred_nLargest.index[index_pred]]
            validateFinalValueError = 100*(maxValueActual-maxValuePred)/maxValueActual

            self.validateFinalValueError.append(validateFinalValueError)

            print('Final Value Error: %+.2f%% for %s' % (self.validateFinalValueError[n], df_validate.columns[n]))

    def ForecastFuture(self):
        # Forecast Future Values
        self.scaler = joblib.load(self.scalerPath)
        self.model = tf.keras.models.load_model(self.modelPath)
        
        df_past = self.df.copy()
        df_past = df_past[self.attributeSelected].iloc[-self.n_steps_in:]
        df_past_transformed = self.scaler.transform(df_past[-self.n_steps_in:])
        df_past_transformed = df_past_transformed.reshape((1, self.n_steps_in, self.n_features))

        future_Y = self.model.predict(df_past_transformed)
        future_Y = self.scaler.inverse_transform(future_Y[0])

        # Change Columns Name of Forecasted Data 
        attributes_Pred = [attribute + "_Prediction" for attribute in self.attributeSelected]
        df_future = pd.DataFrame(future_Y,columns=attributes_Pred)

        # Change Index of Forecasted Data
        firstdate_Forecast = self.df.index[-1] + datetime.timedelta(days=1)
        date_list = [firstdate_Forecast + datetime.timedelta(days=days) for days in range(self.n_steps_out)]
        df_future.index = date_list

        # Add Forecasted Data to Original DataFrame and save as csv file
        df_combined = pd.concat([self.df, df_future], axis=1, sort=False)
        df_combined.index.name = "Date"
        
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.forecastDataFilename = "./service_dist/combined_forecast_data_" + now + ".csv"
        df_combined.to_csv(self.forecastDataFilename)

        forecastDataFile = open(self.forecastDataFilename, mode='r')
        self.forecastData = forecastDataFile.read()
        forecastDataFile.close()

        # df_combined.plot(figsize=(15,6))
        # plt.show()

        # Get Forecasted Data and Compare with Previous Data
        self.maxValuePred = []
        self.percentDifference = []

        for n in range(len(self.attributeSelected)):
            # Get Largest Nth Values from DataFrame for Actual Next Quarter Prediction
            df_nLargest = df_past[-90:].nlargest(10, df_past.columns[n])
            df_future_nLargest = df_future.nlargest(10, df_future.columns[n])

            timediff = np.diff(df_nLargest.index.values.astype(dtype='datetime64[D]')).astype('int')
            timediff_pred = np.diff(df_future_nLargest.index.values.astype(dtype='datetime64[D]')).astype('int')

            index = 0
            index_pred = 0

            if(any(timediff > 30)):
                
                timediff = timediff[:(np.argmax(timediff > 30) + 1)]

                if(any(timediff < -30)):
                    index = np.argmax(timediff < 30)
                else:
                    index = np.argmax(timediff > 30) + 1

            if(any(timediff_pred > 30)):
                
                timediff = timediff_pred[:(np.argmax(timediff_pred > 30) + 1)]

                if(any(timediff_pred < -30)):
                    index_pred = np.argmax(timediff_pred < 30)
                else:
                    index_pred = np.argmax(timediff_pred > 30) + 1
            
            maxValuePrev = df_past[df_past.columns[n]].loc[df_nLargest.index[index]]
            maxValuePred = df_future[df_future.columns[n]].loc[df_future_nLargest.index[index_pred]]
            percentDifference = 100*(maxValuePred - maxValuePrev) / maxValuePrev

            self.maxValuePred.append(maxValuePred)
            self.percentDifference.append(percentDifference)

            print("Forecasted Value for %s is %.2f" % (self.attributeSelected[n], maxValuePred))
            print("Percent Difference vs Past Quarter for %s: %+.2f%%" % (self.attributeSelected[n], percentDifference))

class TrainingProgressCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        observer.trigger('progress', epoch+1)

class ForecastServicer(forecast_pb2_grpc.ForecastAPIServicer):
    
    def RetrainModel(self, request, context):
        
        self.datasetPath_ = request.DatasetPath
        self.n_steps_in_ = request.DaysToTrain
        self.n_steps_out_ = request.DaysForecastAhead
        self.n_epochs_ = request.NumberEpochs
        self.n_neurons_ = request.NumberNeurons
        self.attributeSelected_ = request.ForecastAttributes

        if(request.DatasetPath == ""):
            # self.datasetPath_ = "./Data Science_20200214.xlsx"
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No Dataset File is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        if(request.DaysToTrain == 0):
            # self.n_steps_in_ = 120
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No DaysToTrain is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        if(request.DaysForecastAhead == 0):
            # self.n_steps_out_ = 90
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No DaysToForecast is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        if(request.NumberEpochs == 0):
            # self.n_epochs_ = 300
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No NumberEpochs is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))
        
        if(request.NumberNeurons == 0):
            # self.n_neurons_ = 300
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No NumberNeurons is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        if(request.ForecastAttributes == []):
            # self.attributeSelected_ = ["Shipment", "Order", "Revenue"]
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No Attributes is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        RetrainObj = ForecastModel(attributes=self.attributeSelected_, n_steps_in=self.n_steps_in_, n_steps_out=self.n_steps_out_, n_epochs=self.n_epochs_, n_neurons=self.n_neurons_, datasetPath=self.datasetPath_)
        RetrainObj.LoadDataset()
        RetrainObj.TransformDataset()
        
        trainModelThread = threading.Thread(target=RetrainObj.TrainModel, daemon=True)
        trainModelThread.start()

        global PROGRESS
        n = 0
        while(PROGRESS < (self.n_epochs_ - 1)):
            time.sleep(5)
            if(PROGRESS == n):
                pass
            else:
                n = PROGRESS
                response = forecast_pb2.TrainResponse(Progress=PROGRESS)
                yield response
        
        trainModelThread.join()
        RetrainObj.ValidateModel()

        # Return Response
        finalResponse = forecast_pb2.TrainResponse()
        finalResponse.Progress = PROGRESS
        finalResponse.ValidationRMSE.extend(RetrainObj.validateRMSEScore)
        finalResponse.ValidationFinalError.extend(RetrainObj.validateFinalValueError)
        finalResponse.ValidationValuesFilePath = os.path.realpath(RetrainObj.validateDataFilename)
        finalResponse.ValidationData = (RetrainObj.validateData)

        yield finalResponse
        
    def Forecast(self, request, context):

        self.datasetPath_ = request.DatasetPath
        self.n_steps_in_ = request.DaysToTrain
        self.n_steps_out_ = request.DaysForecastAhead
        self.attributeSelected_ = request.ForecastAttributes
        self.modelPath_ = request.ModelPath
        self.scalerPath_ = request.ScalerPath

        if(request.DatasetPath == ""):
            # self.datasetPath_ = "./Data Science_20200214.xlsx"
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No Dataset File is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        if(request.DaysToTrain == 0):
            # self.n_steps_in_ = 120
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No DaysToTrain is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        if(request.DaysForecastAhead == 0):
            # self.n_steps_out_ = 90
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No DaysToForecast is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        if(request.ForecastAttributes == []):
            # self.attributeSelected_ = ["Shipment", "Order", "Revenue"]
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No Attributes is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))
        
        if(request.ModelPath == ""):
            # self.modelPath_ = "./default_model.h5"
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No Model is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))

        if(request.ScalerPath == ""):
            # self.scalerPath_ = "./default_scaler.pkl"
            error_status = status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message='No Scaler is provided.')
            context.abort_with_status(rpc_status.to_status(error_status))
        
        ForecastObj = ForecastModel(attributes=self.attributeSelected_, n_steps_in=self.n_steps_in_, n_steps_out=self.n_steps_out_, datasetPath=self.datasetPath_, modelPath=self.modelPath_, scalerPath=self.scalerPath_)
        ForecastObj.LoadDataset()
        ForecastObj.ForecastFuture()

        # Return Response 
        response = forecast_pb2.ForecastResponse()

        response.ForecastedValuesFilePath = os.path.realpath(ForecastObj.forecastDataFilename)
        response.ForecastData = (ForecastObj.forecastData)
        response.MaxForecastedValue.extend(ForecastObj.maxValuePred)
        response.PercentDifference.extend(ForecastObj.percentDifference)

        return response

def main():
    forecast_pb2_grpc.add_ForecastAPIServicer_to_server(ForecastServicer(),server)
    server.add_insecure_port('localhost:50051')
    print("Server is running at localhost:50051")
    server.start()
    
def exit_handler():
    server.stop(0)

if __name__== "__main__":
    main()
    atexit.register(exit_handler)
    server.wait_for_termination()