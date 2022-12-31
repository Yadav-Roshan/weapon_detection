import urllib.request
import numpy as np
import onnx
import onnxruntime
import PIL.Image
import streamlit as st
import validators

MAX_DETECTIONS = 64  # Max number of boxes to detect.
PROB_THRESHOLD = 0.20  # Minimum probably to show results.
IOU_THRESHOLD = 0.45


class NonMaxSuppression:
    def __init__(self, max_detections, prob_threshold, iou_threshold):
        self.max_detections = max_detections
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, boxes, class_probs):
        """
        Args:
            boxes (np.array with shape [-1, 4]): bounding boxes. [x, y, x2, y2]
            class_probs: (np.array with shape[-1, num_classes]): probabilities for each boxes and classes.
        """
        assert len(boxes.shape) == 2 and boxes.shape[1] == 4
        assert len(class_probs.shape) == 2
        assert len(boxes) == len(class_probs)
        classes = np.argmax(class_probs, axis=1)
        probs = class_probs[np.arange(len(class_probs)), classes]
        valid_indices = probs > self.prob_threshold
        boxes, classes, probs = boxes[valid_indices, :], classes[valid_indices], probs[valid_indices]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        selected_boxes = []
        selected_classes = []
        selected_probs = []
        max_detections = min(self.max_detections, len(boxes))

        while len(selected_boxes) < max_detections:
            i = np.argmax(probs)
            if probs[i] < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(classes[i])
            selected_probs.append(probs[i])

            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            xy = np.maximum(box[0:2], other_boxes[:, 0:2])
            xy2 = np.minimum(box[2:4], other_boxes[:, 2:4])
            wh = np.maximum(0, xy2 - xy)

            # Calculate Intersection Over Union (IOU)
            overlap_area = wh[:, 0] * wh[:, 1]
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            overlapping_indices = other_indices[np.where(iou > self.iou_threshold)[0]]
            overlapping_indices = np.append(overlapping_indices, i)

            probs[overlapping_indices] = 0

        return np.array(selected_boxes), np.array(selected_classes), np.array(selected_probs)


class Model:
    ANCHORS = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])

    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath))
        assert len(self.session.get_inputs()) == 1
        self.nms = NonMaxSuppression(MAX_DETECTIONS, PROB_THRESHOLD, IOU_THRESHOLD)
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]
        assert len(self.output_names) == 1

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return self._postprocess(outputs[0], self.ANCHORS)

    def _postprocess(self, outputs, anchors):
        assert len(outputs.shape) == 4 and outputs.shape[0] == 1
        outputs = outputs[0].transpose((1, 2, 0))
        assert len(anchors.shape) == 2
        num_anchors = anchors.shape[0]
        height, width, channels = outputs.shape
        assert channels % num_anchors == 0
        num_classes = channels // num_anchors - 5
        outputs = outputs.reshape((height, width, num_anchors, -1))

        x = (outputs[..., 0] + np.arange(width)[np.newaxis, :, np.newaxis]) / width
        y = (outputs[..., 1] + np.arange(height)[:, np.newaxis, np.newaxis]) / height
        w = np.exp(outputs[..., 2]) * anchors[:, 0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[..., 3]) * anchors[:, 1][np.newaxis, np.newaxis, :] / height

        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x, y, x + w, y + h), axis=-1).reshape(-1, 4)

        objectness = self._logistic(outputs[..., 4, np.newaxis])
        class_probs = outputs[..., 5:]
        class_probs = np.exp(class_probs - np.amax(class_probs, axis=3)[..., np.newaxis])
        class_probs = (class_probs / np.sum(class_probs, axis=3)[..., np.newaxis] * objectness).reshape(-1, num_classes)

        detected_boxes, detected_classes, detected_scores = self.nms(boxes, class_probs)
        return {'detected_boxes': detected_boxes.reshape(1, -1, 4), 'detected_classes': detected_classes.reshape(1, -1), 'detected_scores': detected_scores.reshape(1, -1)}

    def _logistic(self, x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    d1 = {0:"Knife", 1:"Pistol"}
    if len(outputs['detected_classes'][0]) > 0:
        return f"A {d1[outputs['detected_classes'][0][0]]} found in the picture."
    else:
        return "No Weapon found in the picture."

def func(ur):
    model = Model("model.onnx")
    z = f"test.{ur[-3:]}"
    urllib.request.urlretrieve(ur,z)
    outputs = model.predict(z)
    return print_outputs(outputs)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://static.wixstatic.com/media/a13558_e66d4dd6166d41759dec3003daead230~mv2.jpeg/v1/fill/w_770,h_550,al_c,q_85,usm_0.66_1.00_0.01/a13558_e66d4dd6166d41759dec3003daead230~mv2.jpeg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

ur = st.text_input("Enter URL of the image: ", placeholder = "Example: https://example.com")
valid = validators.url(ur)
if ur:
    if(valid==False):
        st.write("Please enter a correct URL.")
    else:
        st.write(func(ur))