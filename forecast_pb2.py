# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: forecast.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='forecast.proto',
  package='ForecastService',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0e\x66orecast.proto\x12\x0f\x46orecastService\"\x9c\x01\n\x0cTrainRequest\x12\x13\n\x0b\x44\x61tasetPath\x18\x01 \x01(\t\x12\x13\n\x0b\x44\x61ysToTrain\x18\x02 \x01(\x05\x12\x19\n\x11\x44\x61ysForecastAhead\x18\x03 \x01(\x05\x12\x14\n\x0cNumberEpochs\x18\x04 \x01(\x05\x12\x15\n\rNumberNeurons\x18\x05 \x01(\x05\x12\x1a\n\x12\x46orecastAttributes\x18\x06 \x03(\t\"\x91\x01\n\rTrainResponse\x12\x10\n\x08Progress\x18\x01 \x01(\x02\x12\x16\n\x0eValidationRMSE\x18\x02 \x03(\x02\x12\x1c\n\x14ValidationFinalError\x18\x03 \x03(\x02\x12 \n\x18ValidationValuesFilePath\x18\x04 \x01(\t\x12\x16\n\x0eValidationData\x18\x05 \x01(\t\"\x99\x01\n\x0f\x46orecastRequest\x12\x13\n\x0b\x44\x61tasetPath\x18\x01 \x01(\t\x12\x1a\n\x12\x46orecastAttributes\x18\x02 \x03(\t\x12\x13\n\x0b\x44\x61ysToTrain\x18\x03 \x01(\x05\x12\x19\n\x11\x44\x61ysForecastAhead\x18\x04 \x01(\x05\x12\x12\n\nScalerPath\x18\x05 \x01(\t\x12\x11\n\tModelPath\x18\x06 \x01(\t\"\x81\x01\n\x10\x46orecastResponse\x12\x1a\n\x12MaxForecastedValue\x18\x01 \x03(\x02\x12\x19\n\x11PercentDifference\x18\x02 \x03(\x02\x12 \n\x18\x46orecastedValuesFilePath\x18\x03 \x01(\t\x12\x14\n\x0c\x46orecastData\x18\x04 \x01(\t2\xb3\x01\n\x0b\x46orecastAPI\x12Q\n\x0cRetrainModel\x12\x1d.ForecastService.TrainRequest\x1a\x1e.ForecastService.TrainResponse\"\x00\x30\x01\x12Q\n\x08\x46orecast\x12 .ForecastService.ForecastRequest\x1a!.ForecastService.ForecastResponse\"\x00\x62\x06proto3'
)




_TRAINREQUEST = _descriptor.Descriptor(
  name='TrainRequest',
  full_name='ForecastService.TrainRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='DatasetPath', full_name='ForecastService.TrainRequest.DatasetPath', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='DaysToTrain', full_name='ForecastService.TrainRequest.DaysToTrain', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='DaysForecastAhead', full_name='ForecastService.TrainRequest.DaysForecastAhead', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='NumberEpochs', full_name='ForecastService.TrainRequest.NumberEpochs', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='NumberNeurons', full_name='ForecastService.TrainRequest.NumberNeurons', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ForecastAttributes', full_name='ForecastService.TrainRequest.ForecastAttributes', index=5,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=36,
  serialized_end=192,
)


_TRAINRESPONSE = _descriptor.Descriptor(
  name='TrainResponse',
  full_name='ForecastService.TrainResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Progress', full_name='ForecastService.TrainResponse.Progress', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ValidationRMSE', full_name='ForecastService.TrainResponse.ValidationRMSE', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ValidationFinalError', full_name='ForecastService.TrainResponse.ValidationFinalError', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ValidationValuesFilePath', full_name='ForecastService.TrainResponse.ValidationValuesFilePath', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ValidationData', full_name='ForecastService.TrainResponse.ValidationData', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=195,
  serialized_end=340,
)


_FORECASTREQUEST = _descriptor.Descriptor(
  name='ForecastRequest',
  full_name='ForecastService.ForecastRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='DatasetPath', full_name='ForecastService.ForecastRequest.DatasetPath', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ForecastAttributes', full_name='ForecastService.ForecastRequest.ForecastAttributes', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='DaysToTrain', full_name='ForecastService.ForecastRequest.DaysToTrain', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='DaysForecastAhead', full_name='ForecastService.ForecastRequest.DaysForecastAhead', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ScalerPath', full_name='ForecastService.ForecastRequest.ScalerPath', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ModelPath', full_name='ForecastService.ForecastRequest.ModelPath', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=343,
  serialized_end=496,
)


_FORECASTRESPONSE = _descriptor.Descriptor(
  name='ForecastResponse',
  full_name='ForecastService.ForecastResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='MaxForecastedValue', full_name='ForecastService.ForecastResponse.MaxForecastedValue', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='PercentDifference', full_name='ForecastService.ForecastResponse.PercentDifference', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ForecastedValuesFilePath', full_name='ForecastService.ForecastResponse.ForecastedValuesFilePath', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ForecastData', full_name='ForecastService.ForecastResponse.ForecastData', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=499,
  serialized_end=628,
)

DESCRIPTOR.message_types_by_name['TrainRequest'] = _TRAINREQUEST
DESCRIPTOR.message_types_by_name['TrainResponse'] = _TRAINRESPONSE
DESCRIPTOR.message_types_by_name['ForecastRequest'] = _FORECASTREQUEST
DESCRIPTOR.message_types_by_name['ForecastResponse'] = _FORECASTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrainRequest = _reflection.GeneratedProtocolMessageType('TrainRequest', (_message.Message,), {
  'DESCRIPTOR' : _TRAINREQUEST,
  '__module__' : 'forecast_pb2'
  # @@protoc_insertion_point(class_scope:ForecastService.TrainRequest)
  })
_sym_db.RegisterMessage(TrainRequest)

TrainResponse = _reflection.GeneratedProtocolMessageType('TrainResponse', (_message.Message,), {
  'DESCRIPTOR' : _TRAINRESPONSE,
  '__module__' : 'forecast_pb2'
  # @@protoc_insertion_point(class_scope:ForecastService.TrainResponse)
  })
_sym_db.RegisterMessage(TrainResponse)

ForecastRequest = _reflection.GeneratedProtocolMessageType('ForecastRequest', (_message.Message,), {
  'DESCRIPTOR' : _FORECASTREQUEST,
  '__module__' : 'forecast_pb2'
  # @@protoc_insertion_point(class_scope:ForecastService.ForecastRequest)
  })
_sym_db.RegisterMessage(ForecastRequest)

ForecastResponse = _reflection.GeneratedProtocolMessageType('ForecastResponse', (_message.Message,), {
  'DESCRIPTOR' : _FORECASTRESPONSE,
  '__module__' : 'forecast_pb2'
  # @@protoc_insertion_point(class_scope:ForecastService.ForecastResponse)
  })
_sym_db.RegisterMessage(ForecastResponse)



_FORECASTAPI = _descriptor.ServiceDescriptor(
  name='ForecastAPI',
  full_name='ForecastService.ForecastAPI',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=631,
  serialized_end=810,
  methods=[
  _descriptor.MethodDescriptor(
    name='RetrainModel',
    full_name='ForecastService.ForecastAPI.RetrainModel',
    index=0,
    containing_service=None,
    input_type=_TRAINREQUEST,
    output_type=_TRAINRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Forecast',
    full_name='ForecastService.ForecastAPI.Forecast',
    index=1,
    containing_service=None,
    input_type=_FORECASTREQUEST,
    output_type=_FORECASTRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_FORECASTAPI)

DESCRIPTOR.services_by_name['ForecastAPI'] = _FORECASTAPI

# @@protoc_insertion_point(module_scope)
