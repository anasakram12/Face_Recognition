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
import requests

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
        os.makedirs("static/images", exist_ok=True)
        os.makedirs("static/varient", exist_ok=True)
        os.makedirs("Logs", exist_ok=True)
        
          # API endpoint for logging
        self.log_api_endpoint = "http://192.168.14.102:7578/api/FaceRecognition/Recognize-Logs"
        self.log_api_endpoint_2 = "http://192.168.15.129:5002/add_log"

        
        # Initialize request queue and processing thread
        self.request_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.response_queue = {}  # Store responses for each request
        self.queue_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.queue_thread.start()

    def process_queue(self):
        """
        Continuously process requests from the queue
        """
        while True:
            try:
                # Block and wait for a request
                request_id, image_data = self.request_queue.get()
                
                # Process the image
                result = self.recognize_face(image_data)
                
                # Store the result with the request ID
                self.response_queue[request_id] = result
                
                # Mark task as done
                self.request_queue.task_done()
            
            except Exception as e:
                print(f"Error processing queue: {e}")
                time.sleep(1)  # Prevent tight loop on errors

    def get_recognition_result(self, request_id, timeout=60):
        """
        Retrieve recognition result for a specific request ID
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.response_queue:
                result = self.response_queue.pop(request_id)
                return result
            time.sleep(0.1)
        
        return {"status": "error", "message": "Request processing timed out"}


    def log_recognition_details(self, result):
        """
        Send recognition details to API endpoint
        """
        try:
            # Prepare log data in the required format
            log_entry = {
                "logDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "status": result.get('status', 'Unknown'),
                "inputFilename": result.get('input_filename', ''),
                "inputPath": os.path.join('Get/In', result.get('input_filename', '')),
                "outputFilename": result.get('output_filename', ''),
                "outputPath": os.path.join('Get/Out', result.get('output_filename', '')),
                "confidence": result.get('confidence', 0.0),
                "matchedFilename": result.get('matched_filename', '')
            }

        #     # Make POST request to API
        #     response = requests.post(
        #         self.log_api_endpoint,
        #         json=[log_entry],  # API expects an array of log entries
        #         headers={'Content-Type': 'application/json'}
        #     )

        #     if response.status_code != 200:
        #         print(f"Failed to send log to API. Status code: {response.status_code}")
        #         print(f"Response: {response.text}")

        # except Exception as e:
        #     print(f"Error sending log to API: {e}")

            # List of API endpoints to send the log to
            api_endpoints = [self.log_api_endpoint, self.log_api_endpoint_2]

            # Send log to each endpoint
            for endpoint in api_endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=[log_entry],  # API expects an array of log entries
                        headers={'Content-Type': 'application/json'}
                    )
                    if response.status_code not in [200, 201]:
                        print(f"Failed to send log to API ({endpoint}). Status code: {response.status_code}")
                        print(f"Response: {response.text}")
                    else:
                        print(f"Log sent successfully to API ({endpoint}). Response: {response.text}")
                except Exception as e:
                    print(f"Error sending log to API ({endpoint}): {e}")

        except Exception as e:
            print(f"General error while preparing or sending log: {e}")

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

    def save_unrecognized_face(self, img, face, input_filename):
        # Extract filename without extension as the face ID
        face_id = os.path.splitext(input_filename)[0]
        
        # Create a filename based on the input filename and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unrecognized_{os.path.splitext(input_filename)[0]}_{timestamp}.jpg"
        filepath = os.path.join("static/images", filename)
        
        # Save the image
        cv2.imwrite(filepath, img)
        
        # # Add to database
        # self.face_db[face_id] = {
        #     'embedding': face.embedding,
        #     'image_path': filepath,
        #     'timestamp': timestamp,
        #     'original_input': input_filename
        # }
        
        # # Save updated embeddings
        # self.save_embeddings()
        
        return filepath, face_id

    def _get_next_available_multi_image_key(self, base_identity):
        """
        Find the next available multi-image key for a given identity.
        Returns a key like 'Identity@1', 'Identity@2', etc.
        """
        # Remove any existing @X suffix
        base_identity = base_identity.split('@')[0]
        
        existing_keys = [
            key for key in self.face_db.keys() 
            if key.startswith(f"{base_identity}@")
        ]
        
        if not existing_keys:
            return f"{base_identity}@1"
        
        # Find existing numeric suffixes
        existing_suffixes = [
            int(key.split('@')[1]) 
            for key in existing_keys 
            if len(key.split('@')) > 1 and key.split('@')[1].isdigit()
        ]
        
        # If less than 5 variants, add a new one
        if len(existing_suffixes) < 5:
            return f"{base_identity}@{max(existing_suffixes) + 1}"
        
        # If 5 variants exist, return None to indicate no more storage
        return None

    def _save_multi_image_variant(self, img, base_identity, multi_image_key):
        """
        Save a multi-image variant with the given key.
        """
        if not multi_image_key:
            return None
        
        # Define save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{multi_image_key}.jpg"
        filepath = os.path.join("static/varient", filename)
        
        # Save the image
        cv2.imwrite(filepath, img)
        
        # Add to database
        self.face_db[multi_image_key] = {
            'embedding': self.face_analyzer.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0].embedding,
            'image_path': filepath,
            'timestamp': timestamp,
            'original_input': base_identity
        }
        
        # Save updated embeddings
        self.save_embeddings()
        
        return filepath

    def recognize_face(self, image_data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert base64 to image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Invalid image data"}

        # Save input image
        input_filename = self.save_image(img, "Get/In", "input")

        # Convert to RGB for face analysis
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_analyzer.get(img_rgb)

        if len(faces) == 0:
            # Read and use the placeholder UNRECOGNIZED.png
            unrecognized_img = cv2.imread("Database/UNRECOGNIZED.png")
            if unrecognized_img is None:
                result = {
                    "status": "error",
                    "message": "UNRECOGNIZED.png placeholder not found",
                    "input_filename": input_filename
                }
                self.log_recognition_details(result)
                return result
            
            output_filename = self.save_image(unrecognized_img, "Get/Out", "unrecognized")
            _, buffer = cv2.imencode('.jpg', unrecognized_img)
            unrecognized_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "status": "unrecognized", 
                "message": "No face detected",
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_image": unrecognized_image_base64
            }
            self.log_recognition_details(result)
            return result

        query_embedding = faces[0].embedding

        # Find best match
        best_match = None
        best_similarity = -1
        best_identity = None

        for identity, data in self.face_db.items():
            # Remove @X suffix if present
            base_identity = identity.split('@')[0]
            
            # Calculate similarity
            similarity = self.calculate_similarity(query_embedding, data['embedding'])
            if similarity > best_similarity:
                best_match = data
                best_similarity = similarity
                best_identity = base_identity

        # Process multi-image variant
        multi_image_key = None

        if best_match and best_similarity > self.threshold:
            # Only save variant if similarity is above 0.5
            if best_similarity > 0.5:
                # Try to save multi-image variant
                multi_image_key = self._get_next_available_multi_image_key(best_identity)
                if multi_image_key:
                    self._save_multi_image_variant(img, best_identity, multi_image_key)
            
            # Use the base identity without @X suffix
            base_identity = best_identity.split('@')[0]
            
            # Find the primary image for this identity
            primary_image_path = next(
                (data['image_path'] for identity, data in self.face_db.items() 
                 if identity.split('@')[0] == base_identity and '@' not in identity), 
                None
            )
            
            if not primary_image_path:
                primary_image_path = best_match['image_path']
            
            # Save matched image to Out folder
            matched_image = cv2.imread(primary_image_path)
            output_filename = self.save_image(matched_image, "Get/Out", "match")
            
            # Encode matched image for response
            _, buffer = cv2.imencode('.jpg', matched_image)
            matched_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "status": "matched",
                "confidence": float(best_similarity),
                "matched_image": matched_image_base64,
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_filename": os.path.basename(primary_image_path),
                "multi_image_saved": multi_image_key if best_similarity > 0.5 else None
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
@app.route('/add_image_to_database', methods=['POST'])
def add_image_to_database():
    """
    Add a new face image to the database.
    Expects JSON payload with:
    - 'image': Base64-encoded image data
    - 'identity': A unique identifier for the person (e.g., name or ID)
    """
    try:
        # Check if required fields are present
        if 'image' not in request.json or 'identity' not in request.json:
            return jsonify({"error": "Missing 'image' or 'identity' in request"}), 400

        # Extract data from request
        image_data = request.json['image']
        identity = request.json['identity']

        # Create necessary directories if they don't exist
        os.makedirs("Database/U_images", exist_ok=True)

        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Convert to RGB for face analysis
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_server.face_analyzer.get(img_rgb)

        if len(faces) == 0:
            return jsonify({"error": "No face detected in the image"}), 400

        if len(faces) > 1:
            return jsonify({"error": "Multiple faces detected. Only one face per image is allowed"}), 400

        # Extract embedding
        embedding = faces[0].embedding

        # Generate timestamps and filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save in Database folder with timestamp (original location)
        original_filename = f"{identity}_{timestamp}.jpg"
        original_filepath = os.path.join("Database/U_images", original_filename)
        cv2.imwrite(original_filepath, img)

        # Add to face_db
        face_server.face_db[identity] = {
            'embedding': embedding,
            'image_path': original_filepath,
            'timestamp': timestamp
        }

        # Persist embeddings to file
        face_server.save_embeddings()
        face_server.load_embeddings()

        return jsonify({
            "status": "success",
            "message": f"Image for '{identity}' added to database",
            "filename": original_filename,
            "timestamp": timestamp
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3005)


