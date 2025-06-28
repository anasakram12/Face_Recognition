from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2

# Load images
src_img = cv2.imread("1/4.jpg")
dst_img = cv2.imread("1/2.png")

# Initialize face analyzer
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

# Detect faces
src_faces = app.get(src_img)
dst_faces = app.get(dst_img)

# Load swap model
swapper = get_model('models/inswapper_128.onnx', download=False, ctx_id=0)

# Perform face swap
res = swapper.get(dst_img, dst_faces[0], src_faces[0], paste_back=True)

# Save result
cv2.imwrite("swapped_result.jpg", res)
