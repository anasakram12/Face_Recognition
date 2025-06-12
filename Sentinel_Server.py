from flask import Flask, request, jsonify,send_from_directory
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
from flask_socketio import SocketIO, emit
import json
import eventlet
eventlet.monkey_patch() 

app = Flask(__name__)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='eventlet',  # Use eventlet for better performance with WebSockets
    ping_timeout=10,
    ping_interval=5,
    max_http_buffer_size=50 * 1024 * 1024,  # Increase buffer size for large payloads
    engineio_logger=False,  # Set to True for debugging
    logger=False  # Set to True for debugging
)

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

        self.batch_request_queue = queue.Queue(maxsize=50)  # Queue for batch processing
        self.batch_response_queue = {}  # Store batch processing results
        self.batch_queue_thread = threading.Thread(target=self.process_batch_queue, daemon=True)
        self.batch_queue_thread.start()
        self.recognition_events = []

    def broadcast_recognition_event(self, event_data):
        """
        Broadcast recognition event to connected WebSocket clients
        """
        try:
            # Add to events list (limited to last 100 events)
            self.recognition_events.append(event_data)
            if len(self.recognition_events) > 100:
                self.recognition_events = self.recognition_events[-100:]
            
            # Broadcast via WebSocket using a thread to avoid blocking
            def send_ws_event():
                try:
                    socketio.emit('recognition_event', event_data)
                    print(f"Broadcasted recognition event via WebSocket: {event_data['status']}")
                except Exception as e:
                    print(f"Error emitting WebSocket event: {e}")
            
            # Start a thread to handle the WebSocket emission
            threading.Thread(target=send_ws_event, daemon=True).start()
        except Exception as e:
            print(f"Error broadcasting recognition event: {e}")

    def process_queue(self):
        """
        Continuously process requests from the queue
        """
        while True:
            try:
                # Block and wait for a request
                request_id, image_data , camera_id = self.request_queue.get()
                
                # Process the image
                result = self.recognize_face(image_data, camera_id)
                
                # Store the result with the request ID
                self.response_queue[request_id] = result
                
                # Mark task as done
                self.request_queue.task_done()

                 # Add a small delay to prevent CPU overload
                time.sleep(0.01)       
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


    def log_recognition_details(self, result, camera_id=None):
        """
        Send recognition details to API endpoint and broadcast via WebSocket
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

            # Create WebSocket event data
            ws_event_data = {
                "timestamp": datetime.now().isoformat(),
                "status": result.get('status', 'Unknown'),
                "camera_id": camera_id,
                "camera_name": self.get_camera_name(camera_id),
                "confidence": result.get('confidence', 0.0),
                "matched_filename": result.get('matched_filename', ''),
                "identity": os.path.splitext(result.get('matched_filename', ''))[0] if result.get('matched_filename') else None,
                "input_image": f"/Get/In/{result.get('input_filename', '')}" if result.get('input_filename') else None,
                "matched_image": f"Database/images/{result.get('matched_filename', '')}" if result.get('matched_filename') else None
            }

            # Broadcast via WebSocket
            self.broadcast_recognition_event(ws_event_data)

            # List of API endpoints to send the log to
            api_endpoints = [self.log_api_endpoint, self.log_api_endpoint_2]

            # Send log to each endpoint
            for endpoint in api_endpoints:
                try:
                    # Add camera_id only for log_api_endpoint_2
                    if endpoint == self.log_api_endpoint_2:
                        log_entry["camera_id"] = camera_id

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
    # Add this helper method to the FaceRecognitionServer class
    def get_camera_name(self, camera_id):
        """
        Get a friendly name for a camera ID
        """
        camera_names = {
            1: 'Paiza Gate 1',
            2: 'Paiza Gate 2',
            3: 'Paiza Gate 3',
            4: 'Paiza Gate 4',
            31: 'Membership Counter Out',
            32: 'Membership Counter In',
            61: 'Travel Counter Out',
            62: 'Travel Counter In',
            3701: 'Table 37 Left',
            3702: 'Table 37 Right'
        }
        return camera_names.get(camera_id, f"Camera {camera_id}")
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

    def save_unrecognized_face(self, img, face, input_filename, camera_id=None):
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
        self.load_embeddings()
        
        return filepath

    def recognize_face(self, image_data, camera_id=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert base64 to image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Invalid image data"}

        # Save input image with camera ID if available
        prefix = f"cam{camera_id}_" if camera_id else ""
        input_filename = self.save_image(img, "Get/In", f"{prefix}input")

        # Convert to RGB for face analysis
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_analyzer.get(img_rgb)

        if len(faces) == 0:
            # Read and use the placeholder UNRECOGNIZED.png
            unrecognized_img = cv2.imread("Database/UNRECOGNIZED.png")
            if (unrecognized_img is None):
                result = {
                    "status": "error",
                    "message": "UNRECOGNIZED.png placeholder not found",
                    "input_filename": input_filename
                }
                self.log_recognition_details(result, camera_id)
                return result
            
            # Add camera ID prefix to output filename too
            output_filename = self.save_image(unrecognized_img, "Get/Out", f"{prefix}unrecognized")
            _, buffer = cv2.imencode('.jpg', unrecognized_img)
            unrecognized_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "status": "unrecognized", 
                "message": "No face detected",
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_image": unrecognized_image_base64,
                "camera_id": camera_id
            }
            self.log_recognition_details(result, camera_id)
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
            
            # Save matched image to Out folder with camera ID prefix
            matched_image = cv2.imread(primary_image_path)
            output_filename = self.save_image(matched_image, "Get/Out", f"{prefix}match")
            
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
                "multi_image_saved": multi_image_key if best_similarity > 0.5 else None,
                "camera_id": camera_id
            }
            self.log_recognition_details(result, camera_id)
            return result
        
        # No match found - save the unrecognized face with camera ID prefix if available
        saved_path, face_id = self.save_unrecognized_face(img, faces[0], input_filename, camera_id)
        
        # Read the saved unrecognized face image
        unrecognized_img = cv2.imread(saved_path)
        
        # Save to Out folder with camera ID prefix
        output_filename = self.save_image(unrecognized_img, "Get/Out", f"{prefix}unrecognized")
        
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
            "saved_path": saved_path,
            "camera_id": camera_id
        }
        self.log_recognition_details(result, camera_id)
        return result
        
    def get_top_matches(self, image_data, top_k=3, min_threshold=0.2):
        """
        Get top k matching faces from the database above the minimum threshold.
        
        Args:
            image_data: Base64 encoded image
            top_k: Number of top matches to return (default 3)
            min_threshold: Minimum similarity threshold (default 0.2)
            
        Returns:
            List of dictionaries containing match information
        """
        try:
            # Convert base64 to image
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"status": "error", "message": "Invalid image data"}

            # Convert to RGB for face analysis
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img_rgb)

            if len(faces) == 0:
                return {"status": "error", "message": "No face detected in the image"}

            query_embedding = faces[0].embedding
            matches = []

            # Calculate similarity with all faces in database
            for identity, data in self.face_db.items():
                similarity = self.calculate_similarity(query_embedding, data['embedding'])
                
                # Only consider matches above minimum threshold
                if similarity >= min_threshold:
                    # Remove @X suffix if present for consistent identity matching
                    base_identity = identity.split('@')[0]
                    
                    matches.append({
                        "identity": base_identity,
                        "confidence": float(similarity),
                        "image_path": data['image_path']
                    })

            # Sort matches by confidence and get top k
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            top_matches = matches[:top_k]

            # Read and encode matched images
            for match in top_matches:
                matched_image = cv2.imread(match['image_path'])
                if matched_image is not None:
                    _, buffer = cv2.imencode('.jpg', matched_image)
                    match['matched_image'] = base64.b64encode(buffer).decode('utf-8')
                else:
                    match['matched_image'] = None
                # Remove the file path from response for security
                del match['image_path']

            return {
                "status": "success",
                "matches": top_matches
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def batch_process_faces(self, image_data_list):
        """
        Process multiple faces in batch and return matching results.
        
        Args:
            image_data_list (list): List of base64 encoded image data
            
        Returns:
            list: List of dictionaries containing recognition results for each image
        """
        results = []
        
        for image_data in image_data_list:
            try:
                # Convert base64 to image
                nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    results.append({
                        "status": "error",
                        "message": "Invalid image data"
                    })
                    continue

                # Convert to RGB for face analysis
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.face_analyzer.get(img_rgb)

                if len(faces) == 0:
                    results.append({
                        "status": "unrecognized",
                        "message": "No face detected"
                    })
                    continue

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

                # Check if confidence is above 40%
                if best_match and best_similarity > 0.4:
                    results.append({
                        "status": "matched",
                        "matched_filename": os.path.basename(best_match['image_path']),
                        "confidence": float(best_similarity)
                    })
                else:
                    results.append({
                        "status": "unrecognized",
                        "message": "No match found with confidence above 40%"
                    })

            except Exception as e:
                results.append({
                    "status": "error",
                    "message": str(e)
                })

        return results

    def batch_process_from_folder(self, image_filenames):
        """
        Process multiple images from Get/In folder and return recognition results.
        
        Args:
            image_filenames (list): List of tuples containing (filename, image_id)
            
        Returns:
            list: List of dictionaries containing recognition results for each image
        """
        results = []
        
        for filename, image_id in image_filenames:
            try:
                # Construct full path
                image_path = os.path.join("Get/In", filename)
                
                # Check if file exists
                if not os.path.exists(image_path):
                    results.append({
                        "image_id": image_id,
                        "status": "error",
                        "message": f"File {filename} not found"
                    })
                    continue
                    
                # Read image
                img = cv2.imread(image_path)
                if img is None:
                    results.append({
                        "image_id": image_id,
                        "status": "error",
                        "message": "Invalid image file"
                    })
                    continue

                # Convert to RGB for face analysis
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.face_analyzer.get(img_rgb)

                if len(faces) == 0:
                    results.append({
                        "image_id": image_id,
                        "status": "unrecognized",
                        "message": "No face detected",
                        "filename": filename
                    })
                    continue

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

                # If confidence is above threshold (40%)
                if best_match and best_similarity > 0.4:
                    results.append({
                        "image_id": image_id,
                        "status": "matched",
                        "filename": filename,
                        "confidence": float(best_similarity),
                        "matchedFilename": os.path.basename(best_match['image_path'])
                    })
                else:
                    results.append({
                        "image_id": image_id,
                        "status": "unrecognized",
                        "filename": filename,
                        "message": "No match found with confidence above 40%"
                    })

            except Exception as e:
                results.append({
                    "image_id": image_id,
                    "status": "error",
                    "filename": filename,
                    "message": str(e)
                })

        return results

    # Modify the process_batch_queue method to emit events in real-time
    def process_batch_queue(self):
        """
        Continuously process batch requests from the queue
        """
        while True:
            try:
                # Block and wait for a batch request
                request_id, files_to_process = self.batch_request_queue.get()
                print(f"[BATCH] Received batch request {request_id} with {len(files_to_process)} files")
                
                # Process the batch
                results = []
                for idx, (filename, image_id) in enumerate(files_to_process, 1):
                    try:
                        print(f"[BATCH] Processing {idx}/{len(files_to_process)}: {filename} (ID: {image_id})")
                        
                        # Construct full path
                        image_path = os.path.join("Get/In", filename)
                        
                        if not os.path.exists(image_path):
                            print(f"[BATCH] Error: File not found - {image_path}")
                            result = {
                                "image_id": image_id,
                                "status": "error",
                                "message": f"File {filename} not found"
                            }
                            results.append(result)
                            
                            # Broadcast error event
                            ws_event_data = {
                                "timestamp": datetime.now().isoformat(),
                                "status": "error",
                                "camera_id": None,
                                "camera_name": "Batch Process",
                                "confidence": 0.0,
                                "matched_filename": None,
                                "identity": None,
                                "input_image": None,
                                "matched_image": None,
                                "message": f"File {filename} not found"
                            }
                            self.broadcast_recognition_event(ws_event_data)
                            continue
                        
                        print(f"[BATCH] Reading image: {filename}")    
                        # Read image
                        img = cv2.imread(image_path)
                        if img is None:
                            print(f"[BATCH] Error: Invalid image file - {filename}")
                            result = {
                                "image_id": image_id,
                                "status": "error",
                                "message": "Invalid image file"
                            }
                            results.append(result)
                            
                            # Broadcast error event
                            ws_event_data = {
                                "timestamp": datetime.now().isoformat(),
                                "status": "error",
                                "camera_id": None,
                                "camera_name": "Batch Process",
                                "confidence": 0.0,
                                "matched_filename": None,
                                "identity": None,
                                "input_image": f"/Get/In/{filename}",
                                "matched_image": None,
                                "message": "Invalid image file"
                            }
                            self.broadcast_recognition_event(ws_event_data)
                            continue

                        # Process face recognition
                        print(f"[BATCH] Detecting faces in {filename}")
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        faces = self.face_analyzer.get(img_rgb)

                        if len(faces) == 0:
                            print(f"[BATCH] No faces detected in {filename}")
                            result = {
                                "image_id": image_id,
                                "status": "unrecognized",
                                "message": "No face detected",
                                "filename": filename
                            }
                            results.append(result)
                            
                            # Broadcast unrecognized event
                            ws_event_data = {
                                "timestamp": datetime.now().isoformat(),
                                "status": "unrecognized",
                                "camera_id": None,
                                "camera_name": "Batch Process",
                                "confidence": 0.0,
                                "matched_filename": None,
                                "identity": None,
                                "input_image": f"/Get/In/{filename}",
                                "matched_image": None,
                                "message": "No face detected"
                            }
                            self.broadcast_recognition_event(ws_event_data)
                            continue

                        print(f"[BATCH] Found {len(faces)} faces in {filename}, searching for matches...")
                        # Find best match
                        query_embedding = faces[0].embedding
                        best_match = None
                        best_similarity = -1

                        for identity, data in self.face_db.items():
                            similarity = self.calculate_similarity(query_embedding, data['embedding'])
                            if similarity > best_similarity:
                                best_match = data
                                best_similarity = similarity

                        # If confidence is above threshold (40%)
                        if best_match and best_similarity > 0.4:
                            print(f"[BATCH] Match found for {filename} with confidence {best_similarity:.2f}")
                            result = {
                                "image_id": image_id,
                                "status": "matched",
                                "filename": filename,
                                "confidence": float(best_similarity),
                                "matchedFilename": os.path.basename(best_match['image_path'])
                            }
                            results.append(result)
                            
                            # Broadcast matched event
                            identity = os.path.splitext(os.path.basename(best_match['image_path']))[0]
                            ws_event_data = {
                                "timestamp": datetime.now().isoformat(),
                                "status": "matched",
                                "camera_id": None,
                                "camera_name": "Batch Process",
                                "confidence": float(best_similarity),
                                "matched_filename": os.path.basename(best_match['image_path']),
                                "identity": identity,
                                "input_image": f"/Get/In/{filename}",
                                "matched_image": f"Database/images/{os.path.basename(best_match['image_path'])}"
                            }
                            self.broadcast_recognition_event(ws_event_data)
                        else:
                            print(f"[BATCH] No match found for {filename} (best confidence: {best_similarity:.2f})")
                            result = {
                                "image_id": image_id,
                                "status": "unrecognized",
                                "filename": filename,
                                "message": "No match found with confidence above 40%"
                            }
                            results.append(result)
                            
                            # Broadcast unrecognized event
                            ws_event_data = {
                                "timestamp": datetime.now().isoformat(),
                                "status": "unrecognized",
                                "camera_id": None,
                                "camera_name": "Batch Process",
                                "confidence": best_similarity,
                                "matched_filename": None,
                                "identity": None,
                                "input_image": f"/Get/In/{filename}",
                                "matched_image": None,
                                "message": "No match found with confidence above 40%"
                            }
                            self.broadcast_recognition_event(ws_event_data)
                        
                        # Add a small delay to prevent CPU overload
                        time.sleep(0.01)

                    except Exception as e:
                        print(f"[BATCH] Error processing {filename}: {str(e)}")
                        result = {
                            "image_id": image_id,
                            "status": "error",
                            "filename": filename,
                            "message": str(e)
                        }
                        results.append(result)
                        
                        # Broadcast error event
                        ws_event_data = {
                            "timestamp": datetime.now().isoformat(),
                            "status": "error",
                            "camera_id": None,
                            "camera_name": "Batch Process",
                            "confidence": 0.0,
                            "matched_filename": None,
                            "identity": None,
                            "input_image": f"/Get/In/{filename}",
                            "matched_image": None,
                            "message": str(e)
                        }
                        self.broadcast_recognition_event(ws_event_data)
                
                print(f"[BATCH] Completed batch {request_id}, processed {len(files_to_process)} files")
                # Store the results
                self.batch_response_queue[request_id] = results
                
                # Mark task as done
                self.batch_request_queue.task_done()
                
            except Exception as e:
                print(f"[BATCH] Critical error in batch processing: {e}")
                time.sleep(0.1)  # Reduced sleep time to be more responsive

    def get_batch_result(self, request_id, timeout=36000):
        """
        Retrieve batch processing result for a specific request ID
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.batch_response_queue:
                result = self.batch_response_queue.pop(request_id)
                return result
            time.sleep(0.1)
        
        return {"status": "error", "message": "Batch processing timed out"}

# Initialize server
face_server = FaceRecognitionServer()
# Add these new WebSocket routes
@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')
    # Send last 100 recognition events to newly connected client
    for event in face_server.recognition_events:
        emit('recognition_event', event)
# Add a heartbeat route to help clients check connection status
@app.route('/api/heartbeat', methods=['GET'])
def heartbeat():
    return jsonify({"status": "alive", "timestamp": datetime.now().isoformat()})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from WebSocket')


@app.route('/Get/In/<path:filename>')
def serve_get_in_files(filename):
    get_in_folder = os.path.join(os.getcwd(), 'Get', 'In')
    return send_from_directory(get_in_folder, filename)

@app.route('/Database/images/<path:filename>')
def serve_Database_files(filename):
    get_in_folder = os.path.join(os.getcwd(), 'Database', 'images')
    return send_from_directory(get_in_folder, filename)
# Add a new route to serve the real-time monitoring page
@app.route('/realtime')
def realtime_monitor():
    return app.send_static_file('realtime.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Extract camera_id if it exists in the request
    camera_id = request.json.get('camera_id', None)
    
    try:
        # Add request to queue with camera_id
        face_server.request_queue.put_nowait((request_id, request.json['image'], camera_id))
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
    
@app.route('/add_image_to_database_2', methods=['POST'])
def add_image_to_database_2():
    """
    Add a new face image to the database.
    For multiple faces, selects the largest face with highest confidence.
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

        # If multiple faces, select the best face based on size and confidence
        selected_face = None
        if len(faces) > 1:
            max_score = -1
            for face in faces:
                # Calculate face area
                face_width = face.bbox[2] - face.bbox[0]
                face_height = face.bbox[3] - face.bbox[1]
                face_area = face_width * face_height
                
                # Combine area and detection confidence for overall score
                # Normalize area to 0-1 range assuming max possible area is image size
                normalized_area = face_area / (img.shape[0] * img.shape[1])
                score = (normalized_area * 0.7) + (face.det_score * 0.3)  # Weighted combination
                
                if score > max_score:
                    max_score = score
                    selected_face = face
        else:
            selected_face = faces[0]

        # Extract embedding from selected face
        embedding = selected_face.embedding

        # Generate timestamps and filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save in Database folder with timestamp
        original_filename = f"{identity}.jpg"
        original_filepath = os.path.join("Database/images", original_filename)

        # If multiple faces were detected, add a note to the response
        multiple_faces_detected = len(faces) > 1
        
        # Crop and save only the selected face if multiple faces were detected
        if multiple_faces_detected:
            # Get bounding box of selected face
            x1, y1, x2, y2 = map(int, selected_face.bbox)
            # Add padding around the face (20%)
            padding_x = int((x2 - x1) * 0.2)
            padding_y = int((y2 - y1) * 0.2)
            # Ensure coordinates are within image bounds
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(img.shape[1], x2 + padding_x)
            y2 = min(img.shape[0], y2 + padding_y)
            # Crop image to selected face
            img = img[y1:y2, x1:x2]

        # Save the image
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

        response_data = {
            "status": "success",
            "message": f"Image for '{identity}' added to database",
            "filename": original_filename,
            "timestamp": timestamp
        }

        if multiple_faces_detected:
            response_data["warning"] = "Multiple faces detected - selected largest face with highest confidence"
            response_data["total_faces_detected"] = len(faces)
            response_data["selected_face_confidence"] = float(selected_face.det_score)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500
    
@app.route('/recognize_top_matches', methods=['POST'])
def recognize_top_matches():
    """
    Endpoint to get top 3 matching faces from the database.
    Expects JSON payload with:
    - 'image': Base64-encoded image data
    
    Returns:
    - List of top 3 matches with their identities and confidence scores
    """
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    try:
        # Process the request directly since we need multiple matches
        result = face_server.get_top_matches(
            request.json['image'],
            top_k=3,
            min_threshold=0.2
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_process_files', methods=['POST'])
def batch_process_files():
    """
    Process multiple images from Get/In folder asynchronously.
    Expects JSON payload with:
    - 'files': List of objects containing {filename: string, image_id: string}
    """
    try:
        if 'files' not in request.json:
            return jsonify({"error": "No files provided"}), 400
            
        files_data = request.json['files']
        if not isinstance(files_data, list):
            return jsonify({"error": "Files must be provided as a list"}), 400
            
        # Convert to list of tuples (filename, image_id)
        files_to_process = [(item['filename'], item['image_id']) 
                           for item in files_data 
                           if 'filename' in item and 'image_id' in item]
        
        if not files_to_process:
            return jsonify({"error": "No valid files to process"}), 400
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        try:
            # Add batch request to queue
            face_server.batch_request_queue.put_nowait((request_id, files_to_process))
        except queue.Full:
            return jsonify({"error": "Server is at maximum batch capacity. Please try again later."}), 503
        
        # Wait and retrieve results
        results = face_server.get_batch_result(request_id)
        
        return jsonify({
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create a dedicated thread for batch processing
    batch_thread = threading.Thread(
        target=face_server.process_batch_queue, 
        daemon=True
    )
    batch_thread.start()
    
    # Create a dedicated thread for queue processing
    queue_thread = threading.Thread(
        target=face_server.process_queue,
        daemon=True
    )
    queue_thread.start()
    
    # Use eventlet's WSGI server which is better for WebSockets
    import eventlet.wsgi
    eventlet.wsgi.server(
        eventlet.listen(('0.0.0.0', 3005)), 
        app,
        log_output=False  # Set to True for debugging
    )


