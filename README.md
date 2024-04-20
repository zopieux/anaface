# anaface

Face detection and recognition using onnxruntime

## Installation

```console
$ pip install anaface
```

## Usage

```python
import anaface

analysis = anaface.FaceAnalysis()

faces = analysis.analyze('image.jpeg')

faces = analysis.analyze(PIL.Image.open('image.jpeg'))

faces = analysis.analyze(cv2.imread('image.jpeg'))
```

## License

MIT. Many functions repackaged from https://github.com/deepinsight/insightface, which is MIT.
