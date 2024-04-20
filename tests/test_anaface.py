import pytest
import pathlib
import anaface

THIS = pathlib.Path(__file__).parent
PHOTO = THIS / "pexels-photo-3184398.jpeg"


def test_path():
    faces = anaface.FaceAnalysis().analyze(PHOTO)
    assert len(faces) == 6
    assert sum(1 for f in faces if f.gender == anaface.Gender.MALE) == 4
    assert sum(1 for f in faces if f.gender == anaface.Gender.FEMALE) == 2


def test_pillow():
    from PIL import Image

    faces = anaface.FaceAnalysis().analyze(Image.open(PHOTO))
    assert len(faces) == 6


def test_cv2():
    import cv2

    faces = anaface.FaceAnalysis().analyze(cv2.imread(str(PHOTO)))
    assert len(faces) == 6
