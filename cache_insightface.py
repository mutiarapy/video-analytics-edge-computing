from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('[INFO] InsightFace buffalo_s model files cached to ~/.insightface/')
