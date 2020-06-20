import os
import tensorflow as tf
from string import Template
from PIL import Image
from logging import DEBUG
from pathlib import Path
from threading import Thread, Event
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Queue
from watsor.stream.log import LogHandler
from watsor.stream.work import Work, WorkPublish, Payload
from watsor.stream.share import FrameBuffer
from watsor.stream.sync import CountDownLatch, CountableQueue
from watsor.test.detect_stream import Artist, ShapeDetector

CLASSES = {idx: shape for idx, shape in enumerate(['unlabeled', 'triangle', 'ellipse', 'rectangle'])}

CONFIG = """model {
  ssd {
    num_classes: 3
    image_resizer {
      fixed_shape_resizer {
        height: $height
        width: $width
      }
    }
    feature_extractor {
      type: "ssd_mobilenet_v1"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 3.99999989895e-05
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.0299999993294
          }
        }
        activation: RELU_6
        batch_norm {
          decay: 0.999700009823
          center: true
          scale: true
          epsilon: 0.0010000000475
          train: true
        }
      }
    }
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 3.99999989895e-05
            }
          }
          initializer {
            truncated_normal_initializer {
              mean: 0.0
              stddev: 0.0299999993294
            }
          }
          activation: RELU_6
          batch_norm {
            decay: 0.999700009823
            center: true
            scale: true
            epsilon: 0.0010000000475
            train: true
          }
        }
        min_depth: 0
        max_depth: 0
        num_layers_before_predictor: 0
        use_dropout: false
        dropout_keep_probability: 0.800000011921
        kernel_size: 1
        box_code_size: 4
        apply_sigmoid_to_scores: false
      }
    }
    anchor_generator {
      ssd_anchor_generator {
        num_layers: 6
        min_scale: 0.20000000298
        max_scale: 0.949999988079
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.333299994469
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 0.300000011921
        iou_threshold: 0.600000023842
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_sigmoid {
        }
      }
      hard_example_miner {
        num_hard_examples: 3000
        iou_threshold: 0.990000009537
        loss_type: CLASSIFICATION
        max_negatives_per_positive: 3
        min_negatives_per_image: 0
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
  }
}
train_config {
  batch_size: 24
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    ssd_random_crop {
    }
  }
  optimizer {
    rms_prop_optimizer {
      learning_rate {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.00400000018999
          decay_steps: 800720
          decay_factor: 0.949999988079
        }
      }
      momentum_optimizer_value: 0.899999976158
      decay: 0.899999976158
      epsilon: 1.0
    }
  }
  fine_tune_checkpoint: "$path/training/model.ckpt-XXXX"
  from_detection_checkpoint: true
  num_steps: 200000
}
train_input_reader {
  label_map_path: "$path/annotations/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "$path/annotations/train.record"
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "$path/annotations/label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "$path/annotations/test.record"
  }
}
"""


class Classifier(WorkPublish):

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, frame_buffer,
                 path, group, latch, kwargs=None):
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue, frame_buffer,
                         args=(latch, path, group),
                         kwargs={} if kwargs is None else kwargs)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super(Work, self)._run(stop_event, log_queue, *args, **kwargs)
        try:
            path = args[-2]
            group = args[-1]
            output_path = os.path.join(path, "annotations", "{}.record".format(group))
            with tf.io.TFRecordWriter(output_path) as writer:
                self._spin(self._process, stop_event, *args, writer, **kwargs)

            self._gen_label_map(os.path.join(path, "annotations", "label_map.pbtxt"))
            self._gen_config(os.path.join(path, "ssd.config"), CONFIG, *args, **kwargs)
        except FileNotFoundError as e:
            self._logger.error(e)
        except Exception:
            self._logger.exception('Classification failure')

    def _new_frame(self, frame, payload: Payload, stop_event, frame_buffer: FrameBuffer, latch, path, group, writer,
                   *args, **kwargs):
        try:
            detections = filter(lambda d: d.label > 0, frame.header.detections)

            with Image.frombytes('RGB',
                                 (frame.header.width, frame.header.height),
                                 frame.image.get_obj()) as img:
                count = latch.count_down()
                filename = self._gen_filename(path, group, count + 1, *args, **kwargs)
                img.save(filename)
                self._logger.debug("Frame saved to {}".format(filename))

            tf_example = self._gen_tf_record(frame, detections, filename, *args, **kwargs)
            writer.write(tf_example.SerializeToString())
        finally:
            frame.latch.next()

    @staticmethod
    def _gen_filename(path, group, count, *args, **kwargs):
        return os.path.abspath(os.path.join(path, "images", group, "{:03d}.jpg".format(count)))

    @staticmethod
    def _gen_tf_record(frame, detections, filename, *args, **kwargs):
        width = frame.header.width
        height = frame.header.height
        image_format = b'jpeg'
        with open(filename, "rb") as file:
            encoded_jpg = file.read()
        filename = os.path.basename(filename).encode('utf-8')

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        label = []
        label_text = []
        for detection in detections:
            xmins.append(detection.bounding_box.x_min / width)
            xmaxs.append(detection.bounding_box.x_max / width)
            ymins.append(detection.bounding_box.y_min / height)
            ymaxs.append(detection.bounding_box.y_max / height)
            label.append(detection.label)
            label_text.append(CLASSES.get(detection.label).encode('utf-8'))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=label_text)),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        }))
        return tf_example

    @staticmethod
    def _gen_label_map(path):
        contents = ''
        for idx, shape in CLASSES.items():
            if idx == 0:
                continue

            contents = contents + "item {\n"
            contents = contents + "  id: " + str(idx) + "\n"
            contents = contents + "  name: '" + shape + "'\n}\n\n"

        with open(path, 'w') as f:
            f.write(contents)

    @staticmethod
    def _gen_config(filename, config, frame_queue, stop_event, frame_buffer, *args, **kwargs):
        path = os.path.dirname(filename)
        config = Template(config).substitute(path=path,
                                             width=frame_buffer.frames[0].header.width,
                                             height=frame_buffer.frames[0].header.height)
        os.makedirs(path, exist_ok=True)
        with open(filename, 'w') as f:
            f.write(config)


def prepare_shape_model(groups):
    frame_buffer = FrameBuffer(10, 300, 300)

    frame_queue = Queue(1)
    subscriber_queue = Queue(1)

    log_queue = CountableQueue()
    getLogger().addHandler(QueueHandler(log_queue))

    stop_logging_event = Event()

    log_handler = LogHandler(Thread, "logger", stop_logging_event, log_queue, filename=None)
    log_handler.start()

    for group, count in groups.items():
        path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent.parent, 'build/test/model'))
        os.makedirs(os.path.join(path, "images", group), exist_ok=True)
        os.makedirs(os.path.join(path, "annotations"), exist_ok=True)

        stop_process_event = Event()

        latch = CountDownLatch(count)

        artist = Artist("artist", stop_process_event, log_queue, frame_queue, frame_buffer)
        processes = [artist,
                     ShapeDetector(Thread, "detector", stop_process_event, log_queue, frame_queue, frame_buffer),
                     Classifier(Thread, "classifier", stop_process_event, log_queue, subscriber_queue, frame_buffer,
                                path, group, latch,
                                kwargs={'log_level': DEBUG})]
        artist.subscribe(subscriber_queue)

        for process in processes:
            process.start()

        try:
            latch.wait()
        finally:
            stop_process_event.set()
            for process in processes:
                process.join(30)

    stop_logging_event.set()
    log_queue.join()


if __name__ == '__main__':
    prepare_shape_model({"train": 900, "test": 100})
