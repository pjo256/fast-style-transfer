'''Example streaming ffmpeg numpy processing.

Demonstrates using ffmpeg to decode video input, process the frames in
python, and then encode video output using ffmpeg.

This example uses two ffmpeg processes - one to decode the input video
and one to encode an output video - while the raw frame processing is
done in python with numpy.

At a high level, the signal graph looks like this:

  (input video) -> [ffmpeg process 1] -> [python] -> [ffmpeg process 2] -> (output video)

This example reads/writes video files on the local filesystem, but the
same pattern can be used for other kinds of input/output (e.g. webcam,
rtmp, etc.).

The simplest processing example simply darkens each frame by
multiplying the frame's numpy array by a constant value; see
``process_frame_simple``.

A more sophisticated example processes each frame with tensorflow using
the "deep dream" tensorflow tutorial; activate this mode by calling
the script with the optional `--dream` argument.  (Make sure tensorflow
is installed before running)
'''
from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import argparse
import ffmpeg
import logging
import numpy as np
import os
import subprocess
import zipfile
import tensorflow as tf
import transform
import numpy as np
import vgg
import pdb
import os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy


parser = argparse.ArgumentParser(description='Example streaming ffmpeg numpy processing')
parser.add_argument('in_filename', help='Input filename')
parser.add_argument('out_filename', help='Output filename')
parser.add_argument(
    '--dream', action='store_true', help='Use DeepDream frame processing (requires tensorflow)')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height


def start_ffmpeg_process1(in_filename):
    logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height):
    logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, format='flv', pix_fmt='yuv420p', vcodec='libx264')
        #.output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)


def read_frame(process1, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame


def process_frame_simple(frame):
    '''Simple processing example: darken frame.'''
    return frame * 0.3

def write_frame(process2, frame):
    logger.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )


def run(in_filename, out_filename, process_frame):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            logger.info('End of input stream')
            break

        logger.debug('Processing frame')
        out_frame = process_frame(in_frame)
        write_frame(process2, out_frame)

    logger.info('Waiting for ffmpeg process1')
    process1.wait()

    logger.info('Waiting for ffmpeg process2')
    process2.stdin.close()
    process2.wait()

    logger.info('Done')

class StyleTransfer(object):
    def __init__(self, size, checkpoint_dir, device_t='/gpu:0'):
        self._g = tf.Graph()
        self._soft_config = tf.ConfigProto(allow_soft_placement=True)
        self._soft_config.gpu_options.allow_growth = True
        self._device_t = device_t
        self._g.as_default()
        self._g.device(device_t)
        self._sess = tf.Session(config=self._soft_config)
        self._batch_shape = (1, size[1], size[0], 3)
        self._img_placeholder = tf.placeholder(tf.float32, shape=self._batch_shape,
                                            name='img_placeholder')

        self._preds = transform.net(self._img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(self._sess, checkpoint_dir)

    def process_frame(self, frame, device_t='/gpu:0'):
        X = np.zeros(self._batch_shape, dtype=np.float32)
        X[0] = frame
        _preds = self._sess.run(self._preds, feed_dict={self._img_placeholder: X})
        return _preds

if __name__ == '__main__':
    args = parser.parse_args()
    if args.dream:
        import tensorflow as tf
        size = get_video_size(args.in_filename)
        process_frame = StyleTransfer(size, './models/scream.ckpt').process_frame
    else:
        process_frame = process_frame_simple
    run(args.in_filename, args.out_filename, process_frame)
