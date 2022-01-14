# clients_api
Python clients for inferencing MagCIL Deep Audio API.

Deep Audio API provides access to a set of models that analyze audio signals in terms of:
 - General auditory analysis (discriminate between music, speech, other sounds and silence)
 - Musical classification (musical genres, moods and styles)
 - Speaker characteristics (gender, speaking style etc)
 - Environmental sound analysis (recognize quality of "soundscape")


Access to the API is provided through a simple Python client that sends audio data via an GRPC connection. Audio predictions are returned in a simple json format.



## 0. Install requiremets
```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

You may also need to install portaudio and related dependencies. For ubuntu:
```
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
```

## 1. Client 
In order to request predictions for specific audio files use the following command: 

```python3 client.py -m deployed_model -i <audio_input> -u url -t token --username user```

Where: 
- `deployed_model`: is the name of the deployed ensemble model in server which encapsulates both preprocessing and pytorch models (this argument could be list of models). Select between `ensemble_dali_4_class`, `ensemble_dali_emotions` or both. 
- `audio_input`: is the path of the input wav files to be tested.
- `url`: is the 'url:port' of the grpc server, e.g. localhost:50051
- `token`: is the token to be used for authentication
- `user`: is the email to be used for authentication

## 2. Real-time Client 
In order to request predictions for real-time recorded audios, use the following commad:

```python3 client-real-time.py -m deployed_model -u url -t token --username user``` 

Where: 
- `deployed_model`: is the name of the deployed ensemble model in server which encapsulates both preprocessing and pytorch models (this argument could be list of models).
  Select between 'ensemble_dali_4_class', 'ensemble_dali_emotions' or both.
- `url`: is the 'url:port' of the grpc server, e.g. localhost:50051
- `token`: is the token to be used for authentication
- `user`: is the email to be used for authentication

## 3. Youtube Client 
In order to request prediction for an audio downloaded from a specific youtube url, use the following command:

```python3 client-youtube.py -m deployed_model -i youtube_url -t token -u url --username user```

Where: 
- `deployed_model`: is the name of the deployed ensemble model in server which encapsulates both preprocessing and pytorch models (this argument could be list of models).
  Select between `ensemble_dali_4_class`, `ensemble_dali_emotions` or both.
- `youtube_url` : is the url of the youtube video to be used as audio input.
- `url`: is the 'url:port' of the grpc server, e.g. localhost:50051
- `token`: is the token to be used for authentication
- `user`: is the email to be used for authentication
