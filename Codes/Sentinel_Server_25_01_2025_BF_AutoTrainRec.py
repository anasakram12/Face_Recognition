from flask import Flask, request, jsonify
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import base64
from datetime import datetime
import configparser
import shutil
import uuid
import queue
import threading
import time

app = Flask(__name__)

class FaceRecognitionServer:
    def __init__(self):
        self.config_file = 'config.ini'
        self.config = self.load_config()
        self.threshold = float(self.config.get('Settings', 'threshold', fallback=0.5))
        
        # Initialize face analyzer
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load embeddings
        self.embeddings_file = 'Database/face_embeddings.pkl'
        self.face_db = self.load_embeddings()
        
        # Ensure directories exist
        os.makedirs("Get/In", exist_ok=True)
        os.makedirs("Get/Out", exist_ok=True)
        os.makedirs("Database/UnrecognizedFaces", exist_ok=True)
        os.makedirs("Database/RecognizedFaces", exist_ok=True)
        os.makedirs("Logs", exist_ok=True)
        
        # Define the single log file path
        self.log_file_path = os.path.join("Logs", "face_recognition_log.txt")
        
        # Initialize request queue and processing thread
        self.request_queue = queue.Queue(maxsize=100)
        self.response_queue = {}
        self.queue_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.queue_thread.start()

    def process_queue(self):
        while True:
            try:
                request_id, image_data = self.request_queue.get()
                result = self.recognize_face(image_data)
                self.response_queue[request_id] = result
                self.request_queue.task_done()
            except Exception as e:
                print(f"Error processing queue: {e}")
                time.sleep(1)

    def get_recognition_result(self, request_id, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.response_queue:
                result = self.response_queue.pop(request_id)
                return result
            time.sleep(0.1)
        
        return {"status": "error", "message": "Request processing timed out"}

    def log_recognition_details(self, result):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(self.log_file_path, 'a') as log_file:
            log_file.write("\n\n" + "="*50 + "\n")
            log_file.write(f"Recognition Log: {timestamp}\n")
            log_file.write("="*50 + "\n\n")
            
            log_file.write(f"Status: {result.get('status', 'Unknown')}\n")
            log_file.write(f"Input Filename: {result.get('input_filename', 'N/A')}\n")
            log_file.write(f"Output Filename: {result.get('output_filename', 'N/A')}\n")
            
            if result.get('status') == 'matched':
                log_file.write(f"Confidence: {result.get('confidence', 'N/A')}\n")

    def load_config(self):
        config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            config['Settings'] = {'threshold': '0.5'}
            with open(self.config_file, 'w') as f:
                config.write(f)
        else:
            config.read(self.config_file)
        return config

    def save_embeddings(self):
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump({'embeddings': self.face_db}, f)
        print(f"Saved {len(self.face_db)} embeddings to database.")

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                print(f"Loaded {len(data['embeddings'])} embeddings.")
                return data['embeddings']
        return {}

    def calculate_similarity(self, embedding1, embedding2):
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(embedding1, embedding2))

    def save_image(self, img, folder, prefix):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, img)
        return filename

    def save_recognized_image(self, identity, image):
        """
        Save image for a recognized person, maintaining max 5 images
        Returns original database image path
        """
        # Create folder for the identity if not exists
        identity_folder = os.path.join("Database/RecognizedFaces", identity)
        os.makedirs(identity_folder, exist_ok=True)
        
        # List existing images
        existing_images = sorted([f for f in os.listdir(identity_folder) if f.startswith(f"{identity}_") and f.endswith('.jpg')])
        
        # Determine next image number
        if not existing_images:
            next_num = 1
        else:
            # Get the last number from the most recent file
            last_file = existing_images[-1]
            last_num = int(last_file.split('_')[-1].split('.')[0])
            next_num = last_num + 1 if last_num < 5 else 1
        
        # Filename for new image
        new_filename = f"{identity}_{next_num}.jpg"
        new_filepath = os.path.join(identity_folder, new_filename)
        
        # Save the new image
        cv2.imwrite(new_filepath, image)
        
        # Return original database image path
        original_db_image_path = self.face_db[identity]['image_path']
        
        return original_db_image_path, os.path.basename(original_db_image_path)

    def save_unrecognized_face(self, img, face, input_filename):
        """
        Save unrecognized face to database
        """
        face_id = os.path.splitext(input_filename)[0]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unrecognized_{os.path.splitext(input_filename)[0]}_{timestamp}.jpg"
        filepath = os.path.join("Database/UnrecognizedFaces", filename)
        
        # Save the image
        cv2.imwrite(filepath, img)
        
        # Add to database
        self.face_db[face_id] = {
            'embedding': face.embedding,
            'image_path': filepath,
            'timestamp': timestamp,
            'original_input': input_filename
        }
        
        # Save updated embeddings
        self.save_embeddings()
        
        return filepath, face_id

    def recognize_face(self, image_data):
        """
        Main face recognition method
        """
        # Convert base64 to image
        try:
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"status": "error", "message": f"Image decoding error: {str(e)}"}

        if img is None:
            return {"status": "error", "message": "Invalid image data"}

        # Save input image
        input_filename = self.save_image(img, "Get/In", "input")

        # Convert to RGB for face analysis
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_analyzer.get(img_rgb)

        if len(faces) == 0:
            # Use unrecognized placeholder
            unrecognized_img = cv2.imread("Database/UNRECOGNIZED.png")
            if unrecognized_img is None:
                return {
                    "status": "error",
                    "message": "UNRECOGNIZED.png placeholder not found",
                    "input_filename": input_filename
                }
            
            output_filename = self.save_image(unrecognized_img, "Get/Out", "unrecognized")
            _, buffer = cv2.imencode('.jpg', unrecognized_img)
            unrecognized_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "status": "unrecognized", 
                "message": "No face detected",
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_image": unrecognized_image_base64
            }

        query_embedding = faces[0].embedding

        # Find best match
        best_match = None
        best_similarity = -1
        best_identity = None

        for identity, data in self.face_db.items():
            similarity = self.calculate_similarity(query_embedding, data['embedding'])
            if similarity > best_similarity:
                best_match = data
                best_similarity = similarity
                best_identity = identity

        if best_match and best_similarity > self.threshold:
            # Get the original database image
            original_image_path, original_filename = self.save_recognized_image(best_identity, img)
            
            # Read original database image
            matched_image = cv2.imread(original_image_path)
            
            # Save matched image to Out folder
            output_filename = self.save_image(matched_image, "Get/Out", "match")
            
            # Encode matched image for response
            _, buffer = cv2.imencode('.jpg', matched_image)
            matched_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "status": "matched",
                "confidence": float(best_similarity),
                "matched_image": matched_image_base64,
                "input_filename": input_filename,
                "output_filename": original_filename,  # This will be the original database image filename
                "matched_filename": os.path.basename(best_match['image_path'])
            }
            
            self.log_recognition_details(result)
            return result
        
        # No match found - save the unrecognized face
        saved_path, face_id = self.save_unrecognized_face(img, faces[0], input_filename)
        
        # Read the saved unrecognized face image
        unrecognized_img = cv2.imread(saved_path)
        
        # Save to Out folder
        output_filename = self.save_image(unrecognized_img, "Get/Out", "unrecognized")
        
        # Encode image for response
        _, buffer = cv2.imencode('.jpg', unrecognized_img)
        unrecognized_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result = {
            "status": "unrecognized_saved", 
            "message": "Face not recognized - saved to database",
            "input_filename": input_filename,
            "output_filename": output_filename,
            "matched_image": unrecognized_image_base64,
            "face_id": face_id,
            "saved_path": saved_path
        }
        
        self.log_recognition_details(result)
        return result

# Initialize server
face_server = FaceRecognitionServer()

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    try:
        # Add request to queue
        face_server.request_queue.put_nowait((request_id, request.json['image']))
    except queue.Full:
        return jsonify({"error": "Server is at maximum capacity. Please try again later."}), 503
    
    # Wait and retrieve result
    result = face_server.get_recognition_result(request_id)
    return jsonify(result)

@app.route('/reload_embeddings', methods=['POST'])
def reload_embeddings():
    face_server.face_db = face_server.load_embeddings()
    return jsonify({"status": "success", "message": "Embeddings reloaded"})

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    try:
        new_threshold = float(request.json['threshold'])
        if 0 <= new_threshold <= 1:
            face_server.threshold = new_threshold
            face_server.config.set('Settings', 'threshold', str(new_threshold))
            with open(face_server.config_file, 'w') as f:
                face_server.config.write(f)
            return jsonify({"status": "success", "message": f"Threshold set to {new_threshold}"})
    except (KeyError, ValueError):
        pass
    return jsonify({"error": "Invalid threshold value"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3005)