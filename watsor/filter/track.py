import numpy as np
from scipy.spatial import distance
from collections import defaultdict, deque
from watsor.stream.share import Detection


class TrackFilter(object):
    """Filters recurring detections being observed several times on a row. The parameter 'sensitivity'
    determines how many times a detection has to be observed to be filtered through. Helps to reduce
    false positives.

    This filter also levels flapping of the bounding box around a detected object. The flapping increases
    when different types of detectors process the same video stream. Due to the different object models,
    bounding box even around a stationary object moves constantly. The filter tracks the bounding boxes in
    adjacent frames, and groups together those where centroids are close, considering them belonging to the
    same object.
    """

    def __init__(self, filters=None, sensitivity=5, history=10):
        self.__sensitivity = sensitivity
        self.__history = history
        self.__filters = [] if filters is None else filters
        self.__detections_by_label = defaultdict(list)

    def __call__(self, detections):
        detections = filter(lambda d: d.label > 0 and all(f(d) for f in self.__filters), detections)
        return self._group_and_update(detections)

    def _group_and_update(self, detections):
        # Group detections by label
        groups = defaultdict(list)
        for detection in detections:
            groups[detection.label].append(detection)

        # When sensitivity is greater than 1, the result list of detections may be smaller
        # than the input list and the first findings won't be reported. However we want
        # to perform certain actions like resetting rate limit, even if it is not worth the report.
        suspicious_activity = len(groups) > 0

        # Remove labels that are no longer detected
        for label in list(self.__detections_by_label.keys()):
            if label not in groups:
                for detection_history in self.__detections_by_label[label]:
                    detection_history.clear()
                self.__detections_by_label[label].clear()
                del self.__detections_by_label[label]

        for label, detections in groups.items():
            # Calculate centroids for each input bounding box
            len_bounding_boxes = len(detections)
            input_centroids = np.zeros((len_bounding_boxes, 2), dtype="int")
            for i, detection in enumerate(detections):
                input_centroids[i] = self._centroid(detection.bounding_box)

            # Calculate centroids for each previously found bounding box
            len_existing_centroids = len(self.__detections_by_label[label])
            existing_centroids = np.zeros((len_existing_centroids, 2), dtype="int")
            for i, detection_history in enumerate(self.__detections_by_label[label]):
                existing_centroids[i] = self._centroid(detection_history[0].bounding_box)

            # Compute the distance between each pair of object centroids and input centroids
            distances = distance.cdist(np.array(existing_centroids), input_centroids)
            if len(distances.shape) == 2 and distances.shape[0] > 0 and distances.shape[1] > 0:
                # Find the smallest value in each row and then
                # sort the row indexes in ascending order of their values
                rows = np.argsort(np.amin(distances, axis=1))
                # Find the smallest value in each column and then
                # sort using the previously computed row index list
                cols = np.argmin(distances, axis=1)[rows]
            else:
                rows = []
                cols = []

            # Consider bounding boxes, which centroids are close, belonging to the same object,
            # keep them together.
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                self.__detections_by_label[label][row].append(detections[col])

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len_existing_centroids)).difference(used_rows)
            unused_cols = set(range(len_bounding_boxes)).difference(used_cols)

            # Remove bounding boxes that are no longer detected
            for row in sorted(unused_rows, reverse=True):
                self.__detections_by_label[label][row].clear()
                del self.__detections_by_label[label][row]

            # Add bounding boxes that were not present
            for col in unused_cols:
                history = deque([detections[col]], maxlen=self.__history)
                self.__detections_by_label[label].append(history)

        # For each detection we return a maximum bounding box of all its bounding boxes
        # found within last 10 frames to avoid flapping
        result = []
        for label, detections in self.__detections_by_label.items():
            for detection_history in detections:
                # Skip detection if it's observed less than required number of times
                if len(detection_history) < self.__sensitivity:
                    continue

                detection = self._combine(detection_history)
                result.append(detection)
        return result, suspicious_activity

    @staticmethod
    def _centroid(bounding_box):
        cx = int((bounding_box.x_min + bounding_box.x_max) / 2.0)
        cy = int((bounding_box.y_min + bounding_box.y_max) / 2.0)
        return cx, cy

    @staticmethod
    def _combine(detection_history):
        new_detection = Detection()
        new_detection.label = detection_history[0].label
        new_detection.confidence = detection_history[0].confidence
        new_detection.bounding_box.x_min = detection_history[0].bounding_box.x_min
        new_detection.bounding_box.y_min = detection_history[0].bounding_box.y_min
        new_detection.bounding_box.x_max = detection_history[0].bounding_box.x_max
        new_detection.bounding_box.y_max = detection_history[0].bounding_box.y_max

        for i in range(1, len(detection_history)):
            detection = detection_history[i]
            new_detection.confidence = max(new_detection.confidence, detection.confidence)
            new_detection.bounding_box.x_min = min(new_detection.bounding_box.x_min, detection.bounding_box.x_min)
            new_detection.bounding_box.y_min = min(new_detection.bounding_box.y_min, detection.bounding_box.y_min)
            new_detection.bounding_box.x_max = max(new_detection.bounding_box.x_max, detection.bounding_box.x_max)
            new_detection.bounding_box.y_max = max(new_detection.bounding_box.y_max, detection.bounding_box.y_max)

        zones = set()
        for detection in detection_history:
            for zone in filter(lambda z: z > 0, detection.zones):
                zones.add(zone)

        iterator = iter(zones)
        for zone in range(len(new_detection.zones)):
            try:
                new_detection.zones[zone] = next(iterator)
            except StopIteration:
                new_detection.zones[zone] = 0

        return new_detection
