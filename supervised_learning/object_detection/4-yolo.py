#!/usr/bin/env python3

"""script 1"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    The Yolo class is used for object detection using the YOLOv3 model.
    It initializes with the necessary configurations and loads
    the pre-trained model.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class with the provided model,
        class names, and thresholds.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid_f(self, x):
        """
        Apply the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the YOLO model.
        """

        boxes = []
        for i in range(len(outputs)):
            boxes_i = outputs[i][..., 0:4]
            grid_h_i = outputs[i].shape[0]
            grid_w_i = outputs[i].shape[1]
            anchor_box_i = outputs[i].shape[2]

            for anchor_n in range(anchor_box_i):
                for cy_n in range(grid_h_i):
                    for cx_n in range(grid_w_i):

                        tx_n = outputs[i][cy_n, cx_n, anchor_n, 0:1]
                        ty_n = outputs[i][cy_n, cx_n, anchor_n, 1:2]
                        tw_n = outputs[i][cy_n, cx_n, anchor_n, 2:3]
                        th_n = outputs[i][cy_n, cx_n, anchor_n, 3:4]

                        # size of the anchors
                        pw_n = self.anchors[i][anchor_n][0]
                        ph_n = self.anchors[i][anchor_n][1]

                        # calculating center
                        bx_n = self.sigmoid_f(tx_n) + cx_n
                        by_n = self.sigmoid_f(ty_n) + cy_n

                        # calculating hight and width
                        bw_n = pw_n * np.exp(tw_n)
                        bh_n = ph_n * np.exp(th_n)

                        # generating new center
                        new_bx_n = bx_n / grid_w_i
                        new_by_n = by_n / grid_h_i

                        # generating new hight and width
                        new_bh_n = bh_n / int(self.model.input.shape[2])
                        new_bw_n = bw_n / int(self.model.input.shape[1])

                        # calculating (cx1, cy1) and (cx2, cy2) coords
                        y1 = (new_by_n - (new_bh_n / 2)) * image_size[0]
                        y2 = (new_by_n + (new_bh_n / 2)) * image_size[0]
                        x1 = (new_bx_n - (new_bw_n / 2)) * image_size[1]
                        x2 = (new_bx_n + (new_bw_n / 2)) * image_size[1]

                        boxes_i[cy_n, cx_n, anchor_n, 0] = x1
                        boxes_i[cy_n, cx_n, anchor_n, 1] = y1
                        boxes_i[cy_n, cx_n, anchor_n, 2] = x2
                        boxes_i[cy_n, cx_n, anchor_n, 3] = y2

            boxes.append(boxes_i)

        # 2. box confidence = dim [4]
        confidence = []
        for i in range(len(outputs)):
            confidence_i = self.sigmoid_f(outputs[i][..., 4:5])
            confidence.append(confidence_i)

        # 3. box class_probs = dim [5:]
        probs = []
        for i in range(len(outputs)):
            probs_i = self.sigmoid_f(outputs[i][:, :, :, 5:])
            probs.append(probs_i)

        return (boxes, confidence, probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter and process bounding boxes based on class confidence scores.
        """
        scores = []
        classes = []
        box_classes_scores = []
        index_arg_max = []
        box_classes = []

        for bc_i, probs_j in zip(box_confidences, box_class_probs):
            scores.append(bc_i * probs_j)

        for score in scores:
            index_arg_max = np.argmax(score, axis=-1)

            index_arg_max_flat = index_arg_max.flatten()

            classes.append(index_arg_max_flat)

            score_max = np.max(score, axis=-1)
            score_max_flat = score_max.flatten()
            box_classes_scores.append(score_max_flat)

        boxes = [box.reshape(-1, 4) for box in boxes]

        box_classes = np.concatenate(classes, axis=-1)
        box_classes_scores = np.concatenate(box_classes_scores, axis=-1)

        boxes = np.concatenate(boxes, axis=0)

        filtro = np.where(box_classes_scores >= self.class_t)

        return (boxes[filtro], box_classes[filtro], box_classes_scores[filtro])

    def iou(self, x1, x2, y1, y2, pos1, pos2, area):
        """
        Calculates the Intersection over Union (IoU) between two boxes.
        """

        a = np.maximum(x1[pos1], x1[pos2])
        b = np.maximum(y1[pos1], y1[pos2])

        c = np.minimum(x2[pos1], x2[pos2])
        d = np.minimum(y2[pos1], y2[pos2])

        height = np.maximum(0.0, d - b)
        width = np.maximum(0.0, c - a)

        intersection = width * height
        union = area[pos1] + area[pos2] - intersection
        result = intersection / union

        return result

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Performs Non-Maximum Suppression (NMS) to filter overlapping
        bounding boxes.
        """

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for classes in set(box_classes):
            index = np.where(box_classes == classes)

            filtered = filtered_boxes[index]
            scores = box_scores[index]
            classe = box_classes[index]

            x1 = filtered[:, 0]
            y1 = filtered[:, 1]
            x2 = filtered[:, 2]
            y2 = filtered[:, 3]

            area = (x2 - x1) * (y2 - y1)
            index_list = np.flip(scores.argsort(), axis=0)

            keep = []
            while len(index_list) > 0:
                pos1 = index_list[0]
                pos2 = index_list[1:]
                keep.append(pos1)

                iou = self.iou(x1, x2, y1, y2, pos1, pos2, area)

                below_threshold = np.where(iou <= self.nms_t)[0]
                index_list = index_list[below_threshold + 1]

            # array of piked indexes
            keep = np.array(keep)

            box_predictions.append(filtered[keep])
            predicted_box_classes.append(classe[keep])
            predicted_box_scores.append(scores[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    def load_images(folder_path):
        """
        Loads all images from a specified folder.
        """

        # creating a correct full path argument
        images = []
        image_paths = glob.glob(folder_path + '/*', recursive=False)

        # creating the images list
        for imagepath_i in image_paths:
            images.append(cv2.imread(imagepath_i))

        return (images, image_paths)
