from facenet_pytorch import MTCNN, InceptionResnetV1
MTCNN()
InceptionResnetV1(pretrained='vggface2').eval()
print('[INFO] FaceNet models cached.')