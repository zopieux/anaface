# anaface

Face detection and recognition using onnxruntime for CUDA acceleration.
Also includes gender & age detection.

## Installation

```console
$ pip install https://github.com/zopieux/anaface/releases/download/v1.0.0/anaface-1.0.0-py3-none-any.whl
```

## Usage

```python
import anaface

analysis = anaface.FaceAnalysis()

# From a path.
faces = analysis.analyze('image.jpeg')

# From a Pillow image.
from PIL import Image
faces = analysis.analyze(Image.open('image.jpeg'))

# From a numpy array via OpenCV.
import cv2
faces = analysis.analyze(cv2.imread('image.jpeg'))

faces[0].age              # 31
faces[0].gender           # Gender.MALE
faces[0].bbox             # BBox(coords=(418.45462, 218.93082, 489.2016, 303.84094))
faces[0].detection_score  # 0.90196836
faces[0].embedding.shape  # (512,)
```

## License

MIT. Many functions repackaged from https://github.com/deepinsight/insightface, which is MIT.
