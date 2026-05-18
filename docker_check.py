import torch
import cv2
import numpy
import ultralytics
import flask
import insightface

print('[BUILD] torch      :', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('[BUILD] numpy      :', numpy.__version__)
print('[BUILD] opencv     :', cv2.__version__)
print('[BUILD] ultralytics:', ultralytics.__version__)
print('[BUILD] flask      :', flask.__version__)
print('[BUILD] insightface: OK')
