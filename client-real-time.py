from __future__ import print_function
import argparse
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import logging
import os
import sys
import contextlib
import wave
import grpc
import magcil_api_pb2
import magcil_api_pb2_grpc
import time
from datetime import datetime

"Client of the whole/end-to-end pipeline including both grpc and triton servers"


class GrpcAuth(grpc.AuthMetadataPlugin):
    def __init__(self, key, user):
        self._key = key
        self._user = user

    def __call__(self, context, callback):
        callback((('token', self._key), ('user', self._user),), None)


with open('ssl_keys/ca.crt', 'rb') as fh:
    root_cert = fh.read()
with open('ssl_keys/client.crt', 'rb') as fh:
    client_cert = fh.read() 
with open('ssl_keys/client.key', 'rb') as fh:
    client_key = fh.read()


def make_predictions(stub, fs, filename, dimension, data, models, model_version):
    request = magcil_api_pb2.AudioRequest(filename=filename, dimension=dimension,
                                          data=data, fs=fs, models=models,
                                          model_version=model_version)
    for response in stub.Predict(iter([request])):
        number_of_models = len(response.model_name)
        keys = []
        values = [[] for i in range(number_of_models)]
        for i in range(number_of_models):
            preds = response.preds[i]
            classes = response.classes[i]
            pred_classes = [classes.cl[pred] for pred in preds.p]
            st = 0
            for j in pred_classes:
                et = st + response.step[i]
                values[i].append({"st": st, "et": et, "class": j})
                st += response.step[i]
            keys.append(response.model_name[i])
        dicts = dict(zip(keys, values))
        print(dicts)

def run(models, token, username, model_version="", url='localhost:50051',
        root_certificates=None, private_key=None, certificate_chain=None):
    print(url)
    with grpc.secure_channel(url, grpc.composite_channel_credentials(
        grpc.ssl_channel_credentials(root_certificates=root_cert,
                                     private_key=client_key,
                                     certificate_chain=client_cert),
        grpc.metadata_call_credentials(
            GrpcAuth(token, username)))) as channel:
        try:
            grpc.channel_ready_future(channel).result(timeout=3)
        except grpc.FutureTimeoutError:
            sys.exit('Error connecting to server')
        else:
            stub = magcil_api_pb2_grpc.AudioModelsPredictStub(channel)

            import pyaudio
            import struct

            fs = 8000
            data_3sec = []
            FORMAT = pyaudio.paInt16
            mid_buf_size = int(fs * 1.0)  # 1 sec 
            pa = pyaudio.PyAudio()
            stream = pa.open(format=FORMAT, channels=1, rate=fs,
                             input=True, frames_per_buffer=mid_buf_size)
            count = 0
            while (1):
                count += 1
                block = stream.read(mid_buf_size)
                # number of samples (assuming 16 bit sample resolution)
                count_samples = len(block) / 2  
                # convert byte sequences to list of 16bit samples
                format = "%dh" % (count_samples)
                shorts = struct.unpack(format, block)

                t1 = time.time()

                x = np.array(shorts)
                dimension = x.shape
                x = x.astype('int16')
                now = datetime.now()
                dt_string = now.strftime("%Y_%m_%d__%H_%M_%S")
                filename = f"{dt_string}.wav"

                data = x.tobytes()
                data_3sec += data
                data_3sec = data_3sec[-3 * fs :]
                print(len(data_3sec))
                make_predictions(stub, fs, filename, dimension, data, models, model_version)
                print(time.time() - t1)


if __name__ == '__main__':
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model', nargs='+',
                        type=str,
                        required=True,
                        help='model name'
                        )
    parser.add_argument(
        '-x',
        '--model_version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:50051',
                        help='Inference server URL. Default is localhost:8000')
    parser.add_argument(
        '-rc',
        '--root_certificates',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded root certificates. Default is None.')
    parser.add_argument(
        '-pk',
        '--private_key',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded private key. Default is None.')
    parser.add_argument(
        '-cc',
        '--certificate_chain',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded certicate chain. Default is None.')
    parser.add_argument(
        '-t',
        '--token',
        type=str,
        required=True,
        help='Token')
    parser.add_argument(
        '--username',
        type=str,
        required=True,
        help='Username')

    FLAGS = parser.parse_args()

    run(
        FLAGS.model, FLAGS.token, FLAGS.username, FLAGS.model_version, FLAGS.url,
        root_certificates=FLAGS.root_certificates,
        private_key=FLAGS.private_key,
        certificate_chain=FLAGS.certificate_chain
    )
