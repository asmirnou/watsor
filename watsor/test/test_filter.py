import os
from unittest import TestCase, main
from PIL import Image, ImageDraw
from tempfile import NamedTemporaryFile
from watsor.filter.confidence import ConfidenceFilter
from watsor.filter.area import AreaFilter
from watsor.filter.mask import MaskFilter
from watsor.filter.track import TrackFilter
from watsor.stream.share import Detection, BoundingBox


class TestFilter(TestCase):

    def test_confidence(self):
        confidence_filter = ConfidenceFilter({'detect': [{
            'person': {
                'confidence': 50
            }}]})

        self.assertTrue(confidence_filter(Detection(label=1, confidence=0.70)))
        self.assertFalse(confidence_filter(Detection(label=1, confidence=0.40)))
        self.assertFalse(confidence_filter(Detection(label=2, confidence=0.70)))

    def test_area(self):
        area_filter = AreaFilter({
            'width': 100,
            'height': 100,
            'detect': [{
                'person': {
                    'area': 50
                }}]
        })

        self.assertTrue(area_filter(Detection(label=1, confidence=0.70, bounding_box=BoundingBox(0, 0, 100, 50))))
        self.assertFalse(area_filter(Detection(label=1, confidence=0.70, bounding_box=BoundingBox(0, 0, 50, 50))))
        self.assertFalse(area_filter(Detection(label=2, confidence=0.70, bounding_box=BoundingBox(0, 0, 100, 50))))

    def test_mask(self):
        with self.assertRaisesRegex(AssertionError, "Error reading mask file"):
            MaskFilter({'width': 1, 'height': 1, 'mask': 'notafile.png'})

        tmp_mask_file = NamedTemporaryFile(suffix='.png', delete=False)
        try:
            with Image.new('RGB', (10, 10)) as image:
                image.save(tmp_mask_file.name)

            with self.assertRaisesRegex(AssertionError, "Mask image .+ is not of 32 bit color"):
                MaskFilter({'width': 10, 'height': 10, 'mask': tmp_mask_file.name})

            with Image.new('RGBA', (100, 100)) as image:
                with Image.new("L", image.size) as alpha:
                    draw = ImageDraw.Draw(alpha)
                    draw.rectangle((50, 0, alpha.width, alpha.height), fill=255)
                    image.putalpha(alpha)
                image.save(tmp_mask_file.name)

            with self.assertRaisesRegex(AssertionError, "The size of mask image .+ doesn't match"):
                MaskFilter({'width': 50, 'height': 50, 'mask': tmp_mask_file.name})

            mask_filter = MaskFilter({
                'width': image.width,
                'height': image.height,
                'mask': tmp_mask_file.name,
                'detect': []
            })
        finally:
            # Delete explicitly for compatibility with Windows NT
            tmp_mask_file.close()
            os.unlink(tmp_mask_file.name)

        self.assertFalse(mask_filter(Detection(label=1, confidence=0.70, bounding_box=BoundingBox(20, 20, 40, 80))))
        detection = Detection(label=1, confidence=0.70, bounding_box=BoundingBox(20, 20, 80, 80))
        self.assertTrue(mask_filter(detection))
        self.assertEqual(1, detection.zones[0])

    def test_track(self):
        track_filter = TrackFilter(sensitivity=1, history=2)
        detections, suspicious_activity = track_filter([
            Detection(label=1, confidence=0.70, bounding_box=BoundingBox(50, 50, 60, 60)),
            Detection(label=1, confidence=0.70, bounding_box=BoundingBox(10, 10, 30, 30))
        ])

        self.assertTrue(suspicious_activity)
        self.assertEqual(2, len(detections))
        self.assertListEqual([50, 50, 60, 60], self._to_array(detections[0].bounding_box))
        self.assertListEqual([10, 10, 30, 30], self._to_array(detections[1].bounding_box))

        detections, suspicious_activity = track_filter([
            Detection(label=1, confidence=0.70, bounding_box=BoundingBox(40, 40, 55, 55)),
            Detection(label=1, confidence=0.70, bounding_box=BoundingBox(80, 80, 90, 90))
        ])

        self.assertTrue(suspicious_activity)
        self.assertEqual(2, len(detections))
        self.assertListEqual([40, 40, 60, 60], self._to_array(detections[0].bounding_box))
        self.assertListEqual([80, 80, 90, 90], self._to_array(detections[1].bounding_box))

    @staticmethod
    def _to_array(bounding_box: BoundingBox):
        return [bounding_box.x_min, bounding_box.y_min, bounding_box.x_max, bounding_box.y_max]


if __name__ == '__main__':
    main(verbosity=2)
