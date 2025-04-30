#!/usr/bin/env python3
""" script 0 """
import tensorflow.keras as K


class Yolo:
    """
    The Yolo class is used for object detection using the YOLOv3 model.
    It initializes with the necessary configurations and loads
    the pre-trained model.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Init
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
