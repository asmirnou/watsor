import cv2
import numpy as np
from watsor.config.coco import get_coco_class
from watsor.filter.mask import get_alpha_channel, find_contours


class DrawEffect(object):

    def apply(self, image_in, image_out, shape, header_in, header_out):
        # Draw detections on image
        for detection in filter(lambda d: d.label > 0, header_out.detections):
            coco_class = get_coco_class(detection.label)
            display_str = "{}: {}".format(coco_class.label, "{0:.0%}".format(detection.confidence))
            self._draw(image_out, shape[0],
                       detection.bounding_box.x_min, detection.bounding_box.y_min,
                       detection.bounding_box.x_max, detection.bounding_box.y_max,
                       display_str,
                       coco_class.box_color,
                       coco_class.font_color,
                       coco_class.box_thickness, coco_class.font_thickness,
                       coco_class.font_scale, coco_class.alpha)

    @staticmethod
    def _draw(image, image_height, left, top, right, bottom,
              display_str='', box_color=(0, 255, 0), font_color=(255, 255, 255),
              box_thickness=1, font_thickness=1, font_scale=0.5, alpha=0.55):
        """Adds a bounding box to an image. Bounding box coordinates are specified in absolute (pixel).

        The string passed in display_str is displayed above the bounding box on a rectangle
        filled with the input 'color', blended with the image lying beneath. If the top of the
        bounding box extends to the edge of the image, the string is displayed below the bounding
        box. If the bottom of of the bounding box extends to the edge of the image, the string is
        displayed inside the bounding box.

        Args:
            image (numpy image): image object
            image_height (int): height of the image
            left (int): x_min of bounding box
            top (int): y_min of bounding box
            right (int): x_max of bounding box
            bottom (int): y_max of bounding box
            display_str (str): string to display in box
            box_color (int, int, int): RGB tuple describing color to draw bounding box
            font_color (int, int, int): RGB tuple describing color of the text
            box_thickness (int): line thickness
            font_thickness (int): font thickness
            font_scale (float): font scale factor that is multiplied by the font-specific base size
            alpha (float): opacity of the box with text displayed
        """

        cv2.rectangle(image, (left, top), (right, bottom), box_color, box_thickness)

        if not display_str:
            return

        font = cv2.FONT_HERSHEY_DUPLEX
        size = cv2.getTextSize(display_str, font, font_scale, font_thickness)
        baseline = size[1]
        text_width = size[0][0]
        text_height = size[0][1]

        # Each display_str has a top and bottom margin of 10%
        margin = int(round(np.ceil(0.1 * text_height)))

        # If the total height of the display string added to the top of the bounding
        # box exceeds the top of the image, move the string below the bounding box
        # instead of above, or inside
        total_display_str_height = text_height + 2 * margin
        if top - baseline > total_display_str_height:
            text_bottom = top
        elif bottom + total_display_str_height + baseline < image_height:
            text_bottom = bottom + total_display_str_height + baseline
        else:
            text_bottom = top + total_display_str_height + baseline

        # Blend image with display text box
        pt1 = (left, text_bottom - baseline - text_height - 2 * margin)
        pt2 = (left + text_width + 2 * margin, text_bottom)

        cropped_image = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        solid = np.full(cropped_image.shape, box_color, dtype=np.uint8)
        blended = cv2.addWeighted(cropped_image, alpha, solid, 1 - alpha, 0)
        image[pt1[1]:pt2[1], pt1[0]:pt2[0]] = blended

        cv2.putText(image, display_str,
                    (left + margin, text_bottom - baseline - margin),
                    font, font_scale, font_color, font_thickness, cv2.LINE_AA)


class DrawEffectWithContours(DrawEffect):

    def __init__(self, camera_config):
        alpha_channel, _ = get_alpha_channel(camera_config['mask'],
                                             camera_config['width'],
                                             camera_config['height'])
        self.__contours = find_contours(alpha_channel)

    def apply(self, image_in, image_out, shape, header_in, header_out):
        super().apply(image_in, image_out, shape, header_in, header_out)
        for detection in filter(lambda d: d.label > 0, header_out.detections):
            for zone in filter(lambda z: z > 0, detection.zones):
                cv2.drawContours(image_out, self.__contours, zone - 1, color=(255, 255, 0), thickness=1)
