import importlib.resources
import numpy as np
import numpy.typing
import cv2
import onnx
import onnxruntime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from skimage.transform import SimilarityTransform
from enum import Enum


class Metric(Enum):
    DEFAULT = 1
    MAX = 2


class Gender(Enum):
    FEMALE = 0
    MALE = 1


@dataclass
class BBox:
    """A face bounding box."""

    coords: Tuple[float, float, float, float]

    @property
    def left(self):
        return self.coords[0]

    @property
    def top(self):
        return self.coords[1]

    @property
    def right(self):
        return self.coords[2]

    @property
    def bottom(self):
        return self.coords[3]

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


@dataclass(frozen=False)
class Face:
    """A recognized face."""

    bbox: BBox
    detection_score: float
    kps: Optional[np.array] = None
    embedding: Optional[np.array] = None
    age: Optional[int] = None
    gender_int: Optional[int] = None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return np.linalg.norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def gender(self) -> Optional[Gender]:
        if self.gender_int is None:
            return None
        return Gender(self.gender_int)


class FaceAnalysis:
    """A wrapper around face detection, gender & age detection, and face recognition."""

    def __init__(self, ctx_id: int = 0):
        self.detection = RetinaFace()
        self.detection.prepare(ctx_id)
        self.gender_age = GenderAge()
        self.gender_age.prepare(ctx_id)
        self.recognition = ArcFaceONNX()
        self.recognition.prepare(ctx_id)

    def analyze(
        self,
        path_or_image: Union[str, Path, numpy.typing.NDArray, "PIL.Image"],
        input_size=(640, 640),
        max_faces=None,
        metric=Metric.DEFAULT,
        detection_threshold=0.5,
        nms_threshold=0.4,
    ):
        img = path_or_image
        if isinstance(img, (Path, str)):
            img = cv2.imread(str(img))
        # Handles PIL.Image.
        img = np.array(img)
        bboxes, kpss = self.detection.detect(
            img,
            input_size=input_size,
            max_faces=max_faces,
            metric=metric,
            detection_threshold=detection_threshold,
            nms_threshold=nms_threshold,
        )
        if bboxes.shape[0] == 0:
            return []

        def gen():
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                detection_score = bboxes[i, 4]
                kps = None
                if kpss is not None:
                    kps = kpss[i]
                else:
                    continue
                gender, age = self.gender_age.get_gender_age(img, bbox)
                embedding = self.recognition.get_embedding(img, kps)
                yield Face(
                    bbox=BBox(coords=tuple(bbox)),
                    detection_score=detection_score,
                    kps=kps,
                    embedding=embedding,
                    gender_int=gender,
                    age=age,
                )

        return list(gen())


class _Model:
    def prepare(self, ctx_id):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])


class RetinaFace(_Model):
    """RetinaFace is a face detection model. It finds faces bounding box."""

    model_file_name = "det_10g.onnx"

    def __init__(self):
        # Repacked from MIT https://github.com/deepinsight/insightface.
        _load_session(self)
        self.center_cache = {}
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, img, threshold):
        # Repacked from MIT https://github.com/deepinsight/insightface.
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + self.fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(
        self,
        img,
        input_size=(640, 640),
        max_faces=None,
        metric=Metric.DEFAULT,
        detection_threshold=0.5,
        nms_threshold=0.4,
    ):
        # Repacked from MIT https://github.com/deepinsight/insightface.
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, detection_threshold)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, nms_threshold)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_faces is not None and det.shape[0] > max_faces:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == Metric.MAX:
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_faces]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets, threshold):
        # Repacked from MIT https://github.com/deepinsight/insightface.
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep


def _load_session(self):
    self.model_file = importlib.resources.files("anaface.models").joinpath(
        self.model_file_name
    )
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 4
    self.session = onnxruntime.InferenceSession(self.model_file, opts)


def _load_onnx(self, input_std):
    # Repacked from MIT https://github.com/deepinsight/insightface.
    find_sub = False
    find_mul = False
    for nid, node in enumerate(onnx.load(self.model_file).graph.node[:8]):
        if node.name.startswith("Sub") or node.name.startswith("_minus"):
            find_sub = True
        if node.name.startswith("Mul") or node.name.startswith("_mul"):
            find_mul = True
        if nid < 3 and node.name == "bn_data":
            find_sub = True
            find_mul = True
    if find_sub and find_mul:
        self.input_mean = 0.0
        self.input_std = 1.0
    else:
        self.input_mean = 127.5
        self.input_std = input_std
    input_cfg = self.session.get_inputs()[0]
    self.input_shape = input_cfg.shape
    self.input_name = input_cfg.name
    self.input_size = tuple((self.input_shape)[2:4][::-1])
    outputs = self.session.get_outputs()
    self.output_shape = outputs[0].shape
    self.output_names = [out.name for out in outputs]
    assert len(self.output_names) == 1


class GenderAge(_Model):
    model_file_name = "genderage.onnx"

    def __init__(self):
        _load_session(self)
        _load_onnx(self, 128.0)
        assert self.output_shape[1] == 3

    def get_gender_age(self, img, bbox):
        # Repacked from MIT https://github.com/deepinsight/insightface.
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h) * 1.5)
        aimg, M = transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0]
        assert len(pred) == 3
        gender = np.argmax(pred[:2])
        age = int(np.round(pred[2] * 100))
        return gender, age


class Emotion(_Model):
    """This doesn't work. Always returns 'neutral'."""

    model_file_name = "emotion-ferplus-8.onnx"

    def __init__(self):
        _load_session(self)
        _load_onnx(self, 127.5)
        assert self.output_shape[1] == 8

    def get_emotion(self, img, bbox):
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        scale = self.input_size[0] / (max(w, h) * 1.5)
        aimg, M = transform(img, center, self.input_size[0], scale, rotate)
        gray = (cv2.cvtColor(aimg, cv2.COLOR_RGB2GRAY) / 255.0).astype(np.float32)
        gray = np.array([[gray]])
        pred = self.session.run(self.output_names, {self.input_name: gray})[0]
        assert pred.shape == (1, 8)
        labels = [
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
        ]
        return labels[int(np.argmax(pred[0]))]


class ArcFaceONNX(_Model):
    """ArcFace is a face recognition model. It encodes a face into an 512 dimension embedding."""

    model_file_name = "w600k_r50.onnx"

    def __init__(self):
        _load_session(self)
        _load_onnx(self, 127.5)

    def get_embedding(self, img, kps):
        aimg = norm_crop(img, landmark=kps, image_size=self.input_size[0])
        return self.get_features(aimg).flatten()

    def get_features(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        blob = cv2.dnn.blobFromImages(
            imgs,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        return self.session.run(self.output_names, {self.input_name: blob})[0]

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        return self.session.run(self.output_names, {self.input_name: blob})[0]


def distance2bbox(points, distance, max_shape=None):
    # Repacked from MIT https://github.com/deepinsight/insightface.
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    # Repacked from MIT https://github.com/deepinsight/insightface.
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def estimate_norm(lmk, image_size=112, mode="arcface"):
    # Repacked from MIT https://github.com/deepinsight/insightface.
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    tform = SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    # Repacked from MIT https://github.com/deepinsight/insightface.
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def transform(data, center, output_size, scale, rotation):
    # Repacked from MIT https://github.com/deepinsight/insightface.
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = SimilarityTransform(rotation=rot)
    t4 = SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
