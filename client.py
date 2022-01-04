from __future__ import print_function
import argparse
import numpy as np
import librosa
import logging
import os
import sys
import contextlib
import wave
import grpc
from tqdm import tqdm
import magcil_api_pb2
import magcil_api_pb2_grpc

"Client of the whole/end-to-end pipeline including both grpc and triton servers"

def get_wav_duration(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

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


def run(models, list_of_files, token, username, model_version="", url='localhost:50051',
        root_certificates=None, private_key=None, certificate_chain=None): 
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
            durations = []
            requests = []
            for filename in tqdm(list_of_files):
                durations.append(get_wav_duration(filename))
                x, fs = librosa.load(filename, sr=8000, mono=True)
                dimension = x.shape
                x = (x * (2 ** 15)).astype('int16')
                data = x.tobytes()
                request = magcil_api_pb2.AudioRequest(
                    filename=filename, dimension=dimension,
                    data=data, fs=fs, models=models,
                    model_version=model_version)
                requests.append(request)
            responses = []
            dict_responses = []
            for response in stub.Predict(iter(requests)):
                responses.append(response)
                print("\n --> Predictions for file:", response.filename)
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
                dict_responses.append(dicts)
                print(dicts)
        return responses, durations, dict_responses


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
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input audio / Input folder.')
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

    if os.path.isfile(FLAGS.input) or os.path.isdir(FLAGS.input):
        input_files = FLAGS.input
        if os.path.isfile(input_files):
            list_of_files = [input_files]
        else:
            list_of_files = []
            list_names = os.listdir(input_files)
            for filename in list_names:
                f = os.path.join(input_files, filename)
                list_of_files.append(f)
    else:
        raise Exception("No such file or directory")

    _, _, _ = run(
        FLAGS.model, list_of_files, FLAGS.token, FLAGS.username, FLAGS.model_version, FLAGS.url,
        root_certificates=FLAGS.root_certificates,
        private_key=FLAGS.private_key,
        certificate_chain=FLAGS.certificate_chain
    )

