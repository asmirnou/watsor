import cv2
import math
import numpy as np
import random
from time import time
from PIL import Image, ImageDraw
from collections import namedtuple
from watsor.stream.read import ReadDetectPublish
from watsor.stream.work import Work, WorkPublish, Payload
from watsor.stream.sync import CountDownLatch
from watsor.stream.share import Frame, FrameBuffer

Shape = namedtuple('Shape', ['index', 'bounding_box_area_ratio'])

SHAPE_TRIANGLE = Shape(1, 0.5)
SHAPE_ELLIPSE = Shape(2, 0.8)
SHAPE_RECTANGLE = Shape(3, 1.0)


class Artist(ReadDetectPublish):

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer):
        super().__init__(name, stop_event, log_queue, frame_queue, frame_buffer, args=(stop_event,))

    def _new_frame(self, frame: Frame, *args, **kwargs):
        frame.clear()

        # Create a new image and draw a shape
        with Image.new('RGB', (frame.header.width, frame.header.height)) as image:
            draw = ImageDraw.Draw(image)
            self.draw_random_shapes(image, draw)

            # Copy image to shared buffer
            flat_shape = (frame.header.height * frame.header.width * frame.header.channels,)
            image_np1 = np.array(image, copy=False).reshape(flat_shape)
            image_np2 = np.frombuffer(frame.image.get_obj(), np.uint8)
            np.copyto(image_np2, image_np1)

        frame.header.epoch = time()
        return True

    @classmethod
    def draw_random_shapes(cls, image, draw):
        width = image.width / 2
        height = image.height / 2
        center_x = image.width / 2
        center_y = image.height / 2
        cls._draw_random_shape(image, draw, 0, 0, width, height)
        cls._draw_random_shape(image, draw, center_x, 0, width, height)
        cls._draw_random_shape(image, draw, center_x, center_y, width, height)
        cls._draw_random_shape(image, draw, 0, center_y, width, height)

    @staticmethod
    def _draw_random_shape(image, draw, left, top, width, height):
        fill = (200, 200, 200)

        # Determine shape position
        shape = random.choice([SHAPE_ELLIPSE.index, SHAPE_RECTANGLE.index, SHAPE_TRIANGLE.index])
        p1_x = random.randrange(left, left + math.floor(width / 2))
        p1_y = random.randrange(top, top + math.floor(height / 2))
        p2_x = random.randrange(left + math.ceil(width / 2), left + width)
        p2_y = random.randrange(top + math.ceil(height / 2), top + height)

        # Draw one of three shapes
        if shape == SHAPE_ELLIPSE.index:
            draw.ellipse([(p1_x, p1_y), (p2_x, p2_y)], fill)
        elif shape == SHAPE_RECTANGLE.index:
            draw.rectangle([(p1_x, p1_y), (p2_x, p2_y)], fill)
        elif shape == SHAPE_TRIANGLE.index:
            draw.polygon([(p1_x, p1_y), (p1_x, p2_y), (p2_x, p2_y)], fill)


class ShapeDetector(Work):

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, frame_buffer):
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue,
                         args=(stop_event, frame_buffer))

    def _next_frame(self, payload: Payload, stop_event, frame_buffer: FrameBuffer):
        frame = frame_buffer.frames[payload.frame_index]
        try:
            image_shape, image_np = frame.get_numpy_image(np.uint8)

            # Convert to gray color and find contours
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Detect shape type and bounding box
            c, d = 0, 0
            while c < len(contours) and d < len(frame.header.detections):
                contour = contours[c]
                detection = frame.header.detections[d]
                if self._detect_shape(contour, detection):
                    d += 1
                c += 1
        finally:
            frame.latch.next()

    def _detect_shape(self, contour, detection):
        # Determine bounding box
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            return False

        detection.bounding_box.x_min = x
        detection.bounding_box.y_min = y
        detection.bounding_box.x_max = x + w - 1
        detection.bounding_box.y_max = y + h - 1

        # Detect shape type by comparing the shape area with the area of its bounding box
        ratio = cv2.contourArea(contour) / (w - 1) / (h - 1)

        if abs(ratio - SHAPE_TRIANGLE.bounding_box_area_ratio) <= 0.1:
            detection.confidence = self._calc_confidence(SHAPE_TRIANGLE, ratio)
            detection.label = SHAPE_TRIANGLE.index
        elif abs(ratio - SHAPE_ELLIPSE.bounding_box_area_ratio) <= 0.1:
            detection.confidence = self._calc_confidence(SHAPE_ELLIPSE, ratio)
            detection.label = SHAPE_ELLIPSE.index
        elif abs(ratio - SHAPE_RECTANGLE.bounding_box_area_ratio) <= 0.1:
            detection.confidence = self._calc_confidence(SHAPE_RECTANGLE, ratio)
            detection.label = SHAPE_RECTANGLE.index

        return True

    @staticmethod
    def _calc_confidence(shape, ratio):
        return 100 - abs(ratio - shape.bounding_box_area_ratio)


class ShapeCounter(WorkPublish):

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, frame_buffer, count_down_latch):
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue, frame_buffer,
                         args=(count_down_latch, ))

    def _new_frame(self, frame, payload: Payload, stop_event, frame_buffer: FrameBuffer, count_down_latch: CountDownLatch):
        try:
            for detection in frame.header.detections:
                if detection.label > 0:
                    count_down_latch.count_down()
        finally:
            frame.latch.next()
