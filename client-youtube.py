"""
This script acts as an example client for youtube videos
Usage:
python3 src/client-youtube.py -m <model> -i <youtube_url> -t <token> -u <inference_url>
"""

from __future__ import print_function
import argparse
import youtube_dl
import numpy as np
import client
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import platform


def filter_instance(x):
    cnt_dict = {}
    for inst in x:
        if inst not in cnt_dict:
            cnt_dict[inst] = 1
        else:
            cnt_dict[inst] += 1

    maximum = 0
    pred = x[0]
    for key, value in cnt_dict.items():
        if value > maximum:
            maximum = value
            pred = key
    return [pred for _ in range(len(x))]


def med_filter(signal, frame_size):
    preds = list(signal)
    idx = 0
    signal_len =len(signal)
    while idx + frame_size + 1 <= signal_len:
        preds[idx:(idx + frame_size + 1)] = filter_instance(signal[idx:(idx + frame_size + 1)])
        idx = idx + 1
    return preds


def find_segments(preds):
    segments = []
    start = 0
    previous = ""
    for idx, inst in enumerate(preds):
        if inst == previous:
            idx += 1
        else:
            segments.append((start, idx - 1))
            previous = inst
            start = idx
    filtered_segs = []
    for idx, seg in enumerate(segments):
        if seg[1] - seg[0] >= 5:
            filtered_segs.append(seg)

    return filtered_segs


def get_rect_coords(rect):
    xy = rect.get_xy()
    width = rect.get_width()

    x0 = xy[0]
    x1 = x0 + width
    return x0, x1


def plot_rectangles(y_points):
    start = (0, 0)
    width = 0
    height = 0.6
    previous = -1
    rectangles = []

    for idx, pred in enumerate(y_points):
        if pred == previous:
            width += 1
        else:
            if width >= 5:
                rect = Rectangle(start, width + 1, height,
                                              edgecolor='green',
                                              facecolor='none',
                                              lw=3)
                rectangles.append((start[0], start[0] + width + 1))
                plt.gca().add_patch(rect)
            previous = pred
            width = 0
            start = (idx-0.3, pred-0.3)
    return rectangles


class PlaySound:
    def __init__(self, rectangles):
        self.rectangles = rectangles

    def play(self, event):
        print(1)
        x = event.xdata
        for rect in self.rectangles:
            if rect[0] <= x <= rect[1]:
                start = rect[0] + (rect[1] - rect[0]) / 2 - 2
                os.system(f"ffmpeg -i temp.wav -ss {start} "
                          f"-t 5 temp2.wav -y")

                if platform.system() == "Darwin":  # MacOS
                    os.system("play temp2.wav")
                elif platform.system() == "Linux":
                    os.system("aplay temp2.wav")

        return


def plot_magic(preds):
    classes = set(preds)

    class_dict = {}
    idx = 1
    for inst in classes:
        class_dict[inst] = idx
        idx += 1

    y_points = []
    for pred in preds:
        y_points.append(class_dict[pred])

    fig, ax = plt.subplots()
    ax.scatter(range(len(preds)), y_points, s=15)
    ax.set_facecolor("gainsboro")
    plt.grid()
    plt.yticks(
        ticks=[value for key, value in class_dict.items()],
        labels=[key for key, value in class_dict.items()])

    rectangles = plot_rectangles(y_points)
    sound_player = PlaySound(rectangles)
    fig.canvas.mpl_connect('button_press_event', sound_player.play)

    plt.show()


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

    link = FLAGS.input
    # currently supports only one model
    models = FLAGS.model

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

    for model in models:
        preds = [c["class"] for c in r[0][model]]
        filtered_preds = med_filter(preds, 3)
        plot_magic(filtered_preds)
