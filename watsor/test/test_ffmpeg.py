from unittest import TestCase, installHandler, main
from threading import Thread
from subprocess import PIPE
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Process, Event, Queue
from watsor.stream.log import LogHandler
from watsor.stream.copy import Copy
from watsor.stream.work import WorkPublish
from watsor.stream.share import FrameBuffer
from watsor.stream.sync import CountDownLatch
from watsor.stream.ffmpeg import FFmpegDecoder, FFmpegEncoder
from watsor.test.detect_stream import Artist, ShapeDetector, ShapeCounter


class TestFFmpeg(TestCase):

    def setUp(self):
        """Handle Ctrl+C signal to stop tests prematurely.
        """

        installHandler()

    def test_ffmpeg(self):
        """Runs two FFmpeg subprocesses: the first encodes raw 24-bit frames into raw MPEG-4 video stream,
        the second decodes that stream and feeds the reader again with raw 24-bit frames. The latter triggers
        detection of simple shapes on an image. The number of shapes detected is counted and signals to end
        the test.
        """

        width = 480
        height = 360
        encoder_frame_buffer = FrameBuffer(10, width, height)
        decoder_frame_buffer = FrameBuffer(10, width, height)

        encoder_frame_queue = Queue(1)
        decoder_frame_queue = Queue(1)
        artist_subscribe_queue = Queue(1)
        decoder_subscribe_queue = Queue(1)

        log_queue = Queue()
        getLogger().addHandler(QueueHandler(log_queue))

        stop_process_event = Event()

        latch = CountDownLatch(100)

        encoder = FFmpegEncoder("encoder", stop_process_event, log_queue, encoder_frame_queue, encoder_frame_buffer,
                                ['ffmpeg', '-hide_banner', '-loglevel', 'panic', '-f', 'rawvideo', '-pix_fmt',
                                 'rgb24', '-s', '{}x{}'.format(width, height), '-i', '-', '-an', '-f', 'm4v',
                                 '-'], PIPE)
        decoder = FFmpegDecoder("decoder", stop_process_event, log_queue, decoder_frame_queue, decoder_frame_buffer,
                                ['ffmpeg', '-hide_banner', '-loglevel', 'panic', '-f', 'm4v', '-i', '-', '-f',
                                 'rawvideo', '-pix_fmt', 'rgb24', '-'], PIPE)

        artist = Artist("artist", stop_process_event, log_queue, encoder_frame_queue, encoder_frame_buffer)
        conductor = WorkPublish(Thread, "conductor", stop_process_event, log_queue, artist_subscribe_queue,
                                encoder_frame_buffer)

        processes = [LogHandler(Thread, "logger", stop_process_event, log_queue, filename=None),
                     artist,
                     conductor,
                     encoder,
                     decoder,
                     Copy(Thread, "copier", stop_process_event, log_queue, encoder.stdout, decoder.stdin),
                     ShapeDetector(Process, "detector", stop_process_event, log_queue, decoder_frame_queue,
                                   decoder_frame_buffer),
                     ShapeCounter(Thread, "counter", stop_process_event, log_queue, decoder_subscribe_queue,
                                  decoder_frame_buffer, latch)]

        artist.subscribe(artist_subscribe_queue)
        decoder.subscribe(decoder_subscribe_queue)

        for process in processes:
            process.start()

        try:
            self.assertTrue(latch.wait(15))
        finally:
            stop_process_event.set()
            for process in processes:
                process.join(30)


if __name__ == '__main__':
    main(verbosity=2)
