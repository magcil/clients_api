"""
This script acts as an example client for youtube videos
Usage:
python3 src/client-youtube.py -m <model> -i <youtube_url> -t <token> -u <inference_url>
"""

from __future__ import print_function
import argparse
import numpy as np
import client


if __name__ == '__main__':

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
                        help='Youtube link')
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


    import youtube_dl

    #link = "https://www.youtube.com/watch?v=3A1WFqnvi4k"
    link = FLAGS.input

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=False)
        video_title = info_dict.get('title', None)

    path = f'{video_title}.mp3'
    ydl_opts.update({'outtmpl':path})
    print(path)

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

    import os
    os.system(f"ffmpeg -i \"{path}\" -ar 8000 -ac 1 temp.wav -y")

    _, _, r = client.run(
        FLAGS.model, ["temp.wav"], FLAGS.token, FLAGS.username, FLAGS.model_version, FLAGS.url,
        root_certificates=FLAGS.root_certificates,
        private_key=FLAGS.private_key,
        certificate_chain=FLAGS.certificate_chain
    )

    r_classes = {'4_class': [c['class'] for c in r[0]['4_class']]}
    import plotly.express as px
    fig = px.scatter(r_classes)
    fig.update_traces(marker_size=10)
    fig.show()

