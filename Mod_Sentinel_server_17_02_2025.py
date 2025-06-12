from flask import Flask, request, jsonify, send_from_directory
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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from collections import deque
import gc

# Force eventlet to monkey patch standard library
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=5,
    ping_interval=3,
    max_http_buffer_size=50 * 1024 * 1024,
    engineio_logger=False,
    logger=False
)

class FaceRecognitionServer:
    def __init__(self):
        self.config_file = 'config.ini'
        self.config = self.load_config()
        self.threshold = float(self.config.get('Settings', 'threshold', fallback=0.5))
        
        # Configure number of worker threads based on CPU cores
        self.num_worker_threads = min(16, os.cpu_count() * 2)
        self.num_websocket_workers = 4
        
        # Configure GPU parameters
        self.batch_size = 16  # Process 16 face recognition tasks at once
        
        # Initialize face analyzer
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        # Thread-local storage for face analyzers
        self.thread_local = threading.local()
        
        # Load embeddings
        self.embeddings_file = 'Database/face_embeddings.pkl'
        self.face_db = self.load_embeddings()
        
        # Cache frequently matched faces for faster lookup
        self.frequent_faces_cache = {}
        self.update_frequent_faces_cache()
        
        # Ensure directories exist
        os.makedirs("Get/In", exist_ok=True)
        os.makedirs("Get/Out", exist_ok=True)
        os.makedirs("static/images", exist_ok=True)
        os.makedirs("static/varient", exist_ok=True)
        os.makedirs("Logs", exist_ok=True)
        
        # API endpoint for logging
        self.log_api_endpoint = "http://192.168.14.102:7578/api/FaceRecognition/Recognize-Logs"
        self.log_api_endpoint_2 = "http://192.168.15.129:5002/add_log"

        # Recognition events storage
        self.recognition_events = deque(maxlen=100)  # Use deque with fixed size for better performance
        
        # Initialize request queues with priorities
        self.high_priority_queue = queue.PriorityQueue(maxsize=200)
        self.normal_priority_queue = queue.PriorityQueue(maxsize=500)
        self.batch_request_queue = queue.PriorityQueue(maxsize=50)
        
        # Response storage with expiration
        self.response_cache = {}
        self.batch_response_cache = {}
        
        # WebSocket broadcast queue for non-blocking broadcasts
        self.ws_broadcast_queue = queue.Queue(maxsize=500)
        
        # Tracking metrics
        self.processed_count = 0
        self.start_time = time.time()
        self.request_times = deque(maxlen=1000)  # Track last 1000 request times

        # Set up thread pools
        self.recognition_pool = ThreadPoolExecutor(max_workers=self.num_worker_threads)
        self.batch_processing_pool = ThreadPoolExecutor(max_workers=4)
        self.websocket_pool = ThreadPoolExecutor(max_workers=self.num_websocket_workers)
        
        # Start worker threads
        for _ in range(self.num_worker_threads):
            thread = threading.Thread(target=self.process_recognition_queue, daemon=True)
            thread.start()
        
        # Start batch processing thread
        batch_thread = threading.Thread(target=self.process_batch_queue, daemon=True)
        batch_thread.start()
        
        # Start WebSocket broadcast thread
        for _ in range(self.num_websocket_workers):
            ws_thread = threading.Thread(target=self.process_ws_broadcasts, daemon=True)
            ws_thread.start()
        
        # Start cleanup thread for response cache
        cleanup_thread = threading.Thread(target=self.cleanup_response_cache, daemon=True)
        cleanup_thread.start()
        
        # Start periodic cache update thread
        cache_update_thread = threading.Thread(target=self.periodic_cache_update, daemon=True)
        cache_update_thread.start()

    def update_frequent_faces_cache(self):
        """Update cache of most frequently matched faces"""
        # In a real implementation, you would track frequency of matches
        # For now, we'll just cache all embeddings in a numpy-optimized format
        if not self.face_db:
            return
            
        self.frequent_faces_cache = {}
        for identity, data in self.face_db.items():
            # Check if required fields exist
            if 'embedding' not in data:
                print(f"Warning: Face entry '{identity}' has no embedding, skipping")
                continue
                
            # Normalize the embedding for faster similarity calculation
            normalized_embedding = data['embedding'] / np.linalg.norm(data['embedding'])
            
            # Get optional fields with defaults
            self.frequent_faces_cache[identity] = {
                'normalized_embedding': normalized_embedding,
                'image_path': data.get('image_path', ''),
                'timestamp': data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            }

    def periodic_cache_update(self):
        """Periodically update the face embedding cache"""
        while True:
            time.sleep(60)  # Update cache every minute
            try:
                self.update_frequent_faces_cache()
                # Force garbage collection to free memory
                gc.collect()
            except Exception as e:
                print(f"Error updating cache: {e}")

    def get_face_analyzer(self):
        """Get thread-local face analyzer instance"""
        if not hasattr(self.thread_local, "face_analyzer"):
            # Create a thread-local face analyzer
            self.thread_local.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                root='models',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.thread_local.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        return self.thread_local.face_analyzer

    def cleanup_response_cache(self):
        """Periodically clean up expired responses from cache"""
        while True:
            time.sleep(10)  # Check every 10 seconds
            try:
                current_time = time.time()
                # Clean up recognition response cache
                expired_keys = [
                    key for key, (result, timestamp) in self.response_cache.items()
                    if current_time - timestamp > 60  # 60 seconds expiration
                ]
                for key in expired_keys:
                    if key in self.response_cache:
                        del self.response_cache[key]
                
                # Clean up batch response cache
                batch_expired_keys = [
                    key for key, (result, timestamp) in self.batch_response_cache.items()
                    if current_time - timestamp > 300  # 5 minutes expiration for batch results
                ]
                for key in batch_expired_keys:
                    if key in self.batch_response_cache:
                        del self.batch_response_cache[key]
            except Exception as e:
                print(f"Error in cache cleanup: {e}")

    def process_ws_broadcasts(self):
        """Process WebSocket broadcast events from queue"""
        while True:
            try:
                event_data = self.ws_broadcast_queue.get()
                try:
                    socketio.emit('recognition_event', event_data)
                except Exception as e:
                    print(f"Error emitting WebSocket event: {e}")
                finally:
                    self.ws_broadcast_queue.task_done()
                    
                # Small delay to prevent CPU overload
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in WebSocket broadcast worker: {e}")
                time.sleep(0.1)

    def broadcast_recognition_event(self, event_data):
        """Add recognition event to broadcast queue"""
        try:
            # Add to events list
            self.recognition_events.append(event_data)
            
            # Add to broadcast queue (non-blocking)
            try:
                self.ws_broadcast_queue.put_nowait(event_data)
            except queue.Full:
                # If queue is full, drop oldest event - we prioritize latest events
                try:
                    # Try to get and discard an item first
                    self.ws_broadcast_queue.get_nowait()
                    self.ws_broadcast_queue.task_done()
                    # Then put the new event
                    self.ws_broadcast_queue.put_nowait(event_data)
                except:
                    pass  # If we can't put it in, just move on
        except Exception as e:
            print(f"Error queuing recognition event: {e}")

    def process_recognition_queue(self):
        """Worker thread to process face recognition requests"""
        while True:
            try:
                # Try high priority queue first
                try:
                    _, (request_id, image_data, camera_id) = self.high_priority_queue.get_nowait()
                    priority_queue = self.high_priority_queue
                except queue.Empty:
                    # If high priority queue is empty, try normal priority queue
                    _, (request_id, image_data, camera_id) = self.normal_priority_queue.get(block=True, timeout=0.1)
                    priority_queue = self.normal_priority_queue
                
                # Process request
                start_time = time.time()
                result = self.recognize_face(image_data, camera_id)
                processing_time = time.time() - start_time
                
                # Track request time for metrics
                self.request_times.append(processing_time)
                
                # Store result in cache
                self.response_cache[request_id] = (result, time.time())
                
                # Increment counter
                self.processed_count += 1
                
                # Mark task as done
                priority_queue.task_done()
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)
            except queue.Empty:
                # No requests in queue, just wait a bit
                time.sleep(0.01)
            except Exception as e:
                print(f"Error processing recognition queue: {e}")
                time.sleep(0.1)

    def get_recognition_result(self, request_id, timeout=30):
        """Get recognition result from cache"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.response_cache:
                result, _ = self.response_cache.pop(request_id)
                return result
            time.sleep(0.01)  # Short sleep to reduce CPU usage
        
        return {"status": "error", "message": "Request processing timed out"}

    def log_recognition_details(self, result, camera_id=None):
        """Send recognition details to API endpoint and broadcast via WebSocket"""
        try:
            # Prepare WebSocket event data
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

            # Broadcast via WebSocket (non-blocking)
            self.broadcast_recognition_event(ws_event_data)

            # Handle API logging in a separate thread to not block
            threading.Thread(
                target=self._send_log_to_apis,
                args=(result, camera_id),
                daemon=True
            ).start()
                
        except Exception as e:
            print(f"Error in log_recognition_details: {e}")

    def _send_log_to_apis(self, result, camera_id):
        """Send logs to API endpoints in a separate thread"""
        try:
            # Prepare log data
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

            # List of API endpoints
            api_endpoints = [self.log_api_endpoint, self.log_api_endpoint_2]

            # Send log to each endpoint
            for endpoint in api_endpoints:
                try:
                    # Add camera_id only for log_api_endpoint_2
                    if endpoint == self.log_api_endpoint_2:
                        log_entry["camera_id"] = camera_id

                    # Set a short timeout to avoid blocking
                    response = requests.post(
                        endpoint,
                        json=[log_entry],
                        headers={'Content-Type': 'application/json'},
                        timeout=2.0  # Short timeout
                    )
                except Exception as e:
                    # Just log the error and continue - API logging should not block main processing
                    print(f"Error sending log to API ({endpoint}): {e}")
        except Exception as e:
            print(f"General error while preparing or sending log: {e}")

    def get_camera_name(self, camera_id):
        """Get a friendly name for a camera ID"""
        camera_names = {
            1: 'Paiza Gate 1',
            2: 'Paiza Gate 2',
            3: 'Paiza Gate 3',
            4: 'Paiza Gate 4',
            31: 'Membership Counter Out',
            32: 'Membership Counter In',
            61: 'Travel Counter Out',
            62: 'Travel Counter In',
            1131: 'Membership Counter Out Sup',
            1132: 'Membership Counter In Sup',
            1161: 'Travel Counter Out Sup',
            1162: 'Travel Counter In Sup',
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
        
        # Update the cache when embeddings are saved
        self.update_frequent_faces_cache()

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                print(f"Loaded {len(data['embeddings'])} embeddings.")
                return data['embeddings']
        return {}

    def calculate_similarity(self, embedding1, embedding2):
        """Calculate similarity between two embeddings using optimized method"""
        # Normalize the query embedding
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        
        # If embedding2 is already normalized (from cache), just do the dot product
        return float(np.dot(embedding1, embedding2))

    def save_image(self, img, folder, prefix):
        """Save image with optimized I/O"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        
        # Use threading to save image without blocking
        def save_image_task():
            cv2.imwrite(filepath, img)
        
        # Start a thread to save the image
        threading.Thread(target=save_image_task, daemon=True).start()
        
        return filename

    def save_unrecognized_face(self, img, face, input_filename, camera_id=None):
        """Save unrecognized face with optimized I/O"""
        face_id = os.path.splitext(input_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unrecognized_{os.path.splitext(input_filename)[0]}_{timestamp}.jpg"
        filepath = os.path.join("static/images", filename)
        
        # Use threading to save image without blocking
        def save_image_task():
            cv2.imwrite(filepath, img)
        
        # Start a thread to save the image
        threading.Thread(target=save_image_task, daemon=True).start()
        
        return filepath, face_id

    def _get_next_available_multi_image_key(self, base_identity):
        """Find the next available multi-image key for a given identity"""
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
        """Save a multi-image variant with the given key (optimized I/O)"""
        if not multi_image_key:
            return None
        
        # Define save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{multi_image_key}.jpg"
        filepath = os.path.join("static/varient", filename)
        
        # Extract embedding before we spawn a thread
        embedding = self.face_analyzer.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0].embedding
        
        # Use threading to save image and update database without blocking
        def save_variant_task():
            # Save the image
            cv2.imwrite(filepath, img)
            
            # Add to database
            self.face_db[multi_image_key] = {
                'embedding': embedding,
                'image_path': filepath,
                'timestamp': timestamp,
                'original_input': base_identity
            }
            
            # Update cache for this new face
            normalized_embedding = embedding / np.linalg.norm(embedding)
            self.frequent_faces_cache[multi_image_key] = {
                'normalized_embedding': normalized_embedding,
                'image_path': filepath,
                'timestamp': timestamp
            }
            
            # Save updated embeddings - but do this less frequently
            # to avoid disk I/O bottlenecks
            if len(self.face_db) % 10 == 0:  # Save every 10 new faces
                self.save_embeddings()
        
        # Start a thread to save the image and update database
        threading.Thread(target=save_variant_task, daemon=True).start()
        
        return filepath

    def recognize_face(self, image_data, camera_id=None):
        """Recognize face with optimized performance"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Convert base64 to image
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"status": "error", "message": "Invalid image data"}

            # Prepare camera ID prefix
            prefix = f"cam{camera_id}_" if camera_id else ""
            
            # Save input image - this now happens in a separate thread
            input_filename = self.save_image(img, "Get/In", f"{prefix}input")

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

            # Extract query embedding and normalize it
            query_embedding = faces[0].embedding
            normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Fast vector comparison using numpy operations
            best_match = None
            best_similarity = -1
            best_identity = None

            # Use the cached normalized embeddings for faster comparison
            for identity, data in self.frequent_faces_cache.items():
                # Calculate similarity with pre-normalized embedding
                similarity = float(np.dot(normalized_query_embedding, data['normalized_embedding']))
                if similarity > best_similarity:
                    best_match = {
                        'embedding': query_embedding,  # Keep original for later
                        'image_path': data['image_path'],
                        'timestamp': data['timestamp']
                    }
                    best_similarity = similarity
                    best_identity = identity

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
            
            # No match found - save the unrecognized face
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
            
        except Exception as e:
            # Return meaningful error information
            error_result = {
                "status": "error",
                "message": f"Error during face recognition: {str(e)}",
                "camera_id": camera_id
            }
            return error_result

    def get_top_matches(self, image_data, top_k=3, min_threshold=0.2):
        """Get top k matching faces from the database (optimized)"""
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
            normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Use numpy vectorized operations for faster processing
            similarities = []
            
            # Calculate similarity with all faces in cache
            for identity, data in self.frequent_faces_cache.items():
                similarity = float(np.dot(normalized_query_embedding, data['normalized_embedding']))
                
                # Only consider matches above minimum threshold
                if similarity >= min_threshold:
                    # Remove @X suffix if present for consistent identity matching
                    base_identity = identity.split('@')[0]
                    
                    similarities.append((base_identity, similarity, data['image_path']))

            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:top_k]
            
            # Process matches
            top_matches = []
            for base_identity, similarity, image_path in top_similarities:
                # Read matched image
                matched_image = cv2.imread(image_path)
                matched_image_base64 = None
                
                # Only encode the image if we can read it
                if matched_image is not None:
                    _, buffer = cv2.imencode('.jpg', matched_image)
                    matched_image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                top_matches.append({
                    "identity": base_identity,
                    "confidence": float(similarity),
                    "matched_image": matched_image_base64
                })

            return {
                "status": "success",
                "matches": top_matches
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def process_batch_queue(self):
        """Optimized batch queue processor"""
        while True:
            try:
                # Get a batch request from the queue
                _, (request_id, files_to_process) = self.batch_request_queue.get()
                print(f"[BATCH] Received batch request {request_id} with {len(files_to_process)} files")
                
                # Process the files in smaller batches to optimize GPU throughput
                results = []
                batch_size = 8  # Process 8 files at a time
                
                for i in range(0, len(files_to_process), batch_size):
                    # Get the next batch of files
                    current_batch = files_to_process[i:i+batch_size]
                    
                    # Use ThreadPoolExecutor to process files in parallel
                    futures = []
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        for filename, image_id in current_batch:
                            # Submit task to thread pool
                            futures.append(
                                executor.submit(self._process_batch_file, filename, image_id)
                            )
                        
                        # Collect results as they complete
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                result = future.result()
                                results.append(result)
                                
                                # Broadcast event in real-time
                                if 'ws_event_data' in result:
                                    # Extract and remove the event data
                                    ws_event_data = result.pop('ws_event_data')
                                    self.broadcast_recognition_event(ws_event_data)
                            except Exception as e:
                                print(f"[BATCH] Error processing batch file: {e}")
                    
                    # Add a small delay between batches to prevent GPU overload
                    time.sleep(0.1)
                
                # Store the complete results
                self.batch_response_cache[request_id] = (results, time.time())
                
                # Mark task as done
                self.batch_request_queue.task_done()
                
            except Exception as e:
                print(f"[BATCH] Critical error in batch processing: {e}")
                time.sleep(0.1)

    def _process_batch_file(self, filename, image_id):
        """Process a single file from a batch (optimized for parallelism)"""
        try:
            # Construct full path
            image_path = os.path.join("Get/In", filename)
            
            if not os.path.exists(image_path):
                print(f"[BATCH] Error: File not found - {image_path}")
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
                
                return {
                    "image_id": image_id,
                    "status": "error",
                    "message": f"File {filename} not found",
                    "ws_event_data": ws_event_data
                }
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"[BATCH] Error: Invalid image file - {filename}")
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
                
                return {
                    "image_id": image_id,
                    "status": "error",
                    "message": "Invalid image file",
                    "ws_event_data": ws_event_data
                }

            # Process face recognition
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use thread-local face analyzer to avoid contention
            face_analyzer = self.get_face_analyzer()
            faces = face_analyzer.get(img_rgb)

            if len(faces) == 0:
                print(f"[BATCH] No faces detected in {filename}")
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
                
                return {
                    "image_id": image_id,
                    "status": "unrecognized",
                    "message": "No face detected",
                    "filename": filename,
                    "ws_event_data": ws_event_data
                }

            # Find best match
            query_embedding = faces[0].embedding
            normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            best_match = None
            best_similarity = -1
            best_identity = None
            
            # Use the cached normalized embeddings for faster comparison
            for identity, data in self.frequent_faces_cache.items():
                # Calculate similarity with pre-normalized embedding
                similarity = float(np.dot(normalized_query_embedding, data['normalized_embedding']))
                if similarity > best_similarity:
                    best_match = {
                        'image_path': data['image_path']
                    }
                    best_similarity = similarity
                    best_identity = identity

            # If confidence is above threshold (40%)
            if best_match and best_similarity > 0.4:
                print(f"[BATCH] Match found for {filename} with confidence {best_similarity:.2f}")
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
                
                return {
                    "image_id": image_id,
                    "status": "matched",
                    "filename": filename,
                    "confidence": float(best_similarity),
                    "matchedFilename": os.path.basename(best_match['image_path']),
                    "ws_event_data": ws_event_data
                }
            else:
                print(f"[BATCH] No match found for {filename} (best confidence: {best_similarity:.2f})")
                
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
                
                return {
                    "image_id": image_id,
                    "status": "unrecognized",
                    "filename": filename,
                    "message": "No match found with confidence above 40%",
                    "ws_event_data": ws_event_data
                }
                
        except Exception as e:
            print(f"[BATCH] Error processing {filename}: {str(e)}")
            
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
            
            return {
                "image_id": image_id,
                "status": "error",
                "filename": filename,
                "message": str(e),
                "ws_event_data": ws_event_data
            }

    def get_batch_result(self, request_id, timeout=36000):
        """Get batch processing result from cache"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.batch_response_cache:
                result, _ = self.batch_response_cache.pop(request_id)
                return result
            time.sleep(0.1)
        
        return {"status": "error", "message": "Batch processing timed out"}
    
    def get_performance_metrics(self):
        """Get performance metrics for monitoring"""
        uptime = time.time() - self.start_time
        
        # Calculate requests per second
        rps = self.processed_count / uptime if uptime > 0 else 0
        
        # Calculate average processing time
        avg_processing_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        
        # Calculate queue sizes
        high_priority_size = self.high_priority_queue.qsize()
        normal_priority_size = self.normal_priority_queue.qsize()
        batch_queue_size = self.batch_request_queue.qsize()
        ws_queue_size = self.ws_broadcast_queue.qsize()
        
        return {
            "uptime_seconds": uptime,
            "processed_requests": self.processed_count,
            "requests_per_second": rps,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "high_priority_queue_size": high_priority_size,
            "normal_priority_queue_size": normal_priority_size,
            "batch_queue_size": batch_queue_size,
            "websocket_queue_size": ws_queue_size,
            "recognition_events_count": len(self.recognition_events),
            "face_db_size": len(self.face_db),
            "frequent_faces_cache_size": len(self.frequent_faces_cache),
            "response_cache_size": len(self.response_cache),
            "batch_response_cache_size": len(self.batch_response_cache)
        }

# Initialize server
face_server = FaceRecognitionServer()

# WebSocket routes
@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')
    # Send last 100 recognition events asynchronously
    def send_events():
        for event in list(face_server.recognition_events):
            try:
                emit('recognition_event', event)
                # Small delay between events to prevent overload
                eventlet.sleep(0.01)
            except Exception as e:
                print(f"Error sending event history: {e}")
                break
    
    # Start a greenlet to send events without blocking
    eventlet.spawn(send_events)


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from WebSocket')

# API routes
@app.route('/api/heartbeat', methods=['GET'])
def heartbeat():
    """Lightweight heartbeat endpoint for monitoring"""
    return jsonify({
        "status": "alive", 
        "timestamp": datetime.now().isoformat(),
        "rps": face_server.processed_count / (time.time() - face_server.start_time) 
            if time.time() - face_server.start_time > 0 else 0
    })

@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Get performance metrics"""
    return jsonify(face_server.get_performance_metrics())

@app.route('/Get/In/<path:filename>')
def serve_get_in_files(filename):
    get_in_folder = os.path.join(os.getcwd(), 'Get', 'In')
    return send_from_directory(get_in_folder, filename)

@app.route('/Database/images/<path:filename>')
def serve_Database_files(filename):
    get_in_folder = os.path.join(os.getcwd(), 'Database', 'images')
    return send_from_directory(get_in_folder, filename)

@app.route('/realtime')
def realtime_monitor():
    return app.send_static_file('realtime.html')

@app.route('/Check1')
def Check1_monitor():
    return app.send_static_file('Check1.html')

@app.route('/Check2')
def Check2_monitor():
    return app.send_static_file('Check2.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    """Recognize a face with priority queuing"""
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Extract optional parameters
    camera_id = request.json.get('camera_id')
    priority = request.json.get('priority', 'normal')  # Default to normal priority
    
    try:
        # Add to appropriate queue based on priority
        if priority == 'high':
            # Use current timestamp for priority (lower number = higher priority)
            face_server.high_priority_queue.put_nowait((time.time(), (request_id, request.json['image'], camera_id)))
        else:
            face_server.normal_priority_queue.put_nowait((time.time(), (request_id, request.json['image'], camera_id)))
    except queue.Full:
        return jsonify({
            "error": "Server is at maximum capacity. Please try again later.",
            "queue_sizes": {
                "high_priority": face_server.high_priority_queue.qsize(),
                "normal_priority": face_server.normal_priority_queue.qsize()
            }
        }), 503
    
    # Wait and retrieve result with timeout
    result = face_server.get_recognition_result(request_id)
    return jsonify(result)

@app.route('/reload_embeddings', methods=['POST'])
def reload_embeddings():
    """Reload face embeddings from disk"""
    face_server.face_db = face_server.load_embeddings()
    # Update the cache
    face_server.update_frequent_faces_cache()
    return jsonify({"status": "success", "message": "Embeddings reloaded"})

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """Update recognition threshold"""
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
    """Add a new face image to the database"""
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
        
        # Save image
        cv2.imwrite(original_filepath, img)

        # Add to face_db
        face_server.face_db[identity] = {
            'embedding': embedding,
            'image_path': original_filepath,
            'timestamp': timestamp
        }

        # Update face cache
        normalized_embedding = embedding / np.linalg.norm(embedding)
        face_server.frequent_faces_cache[identity] = {
            'normalized_embedding': normalized_embedding,
            'image_path': original_filepath,
            'timestamp': timestamp
        }

        # Persist embeddings to file
        face_server.save_embeddings()

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
    """Add a new face image to the database with multiple face handling"""
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

        # Update face cache
        normalized_embedding = embedding / np.linalg.norm(embedding)
        face_server.frequent_faces_cache[identity] = {
            'normalized_embedding': normalized_embedding,
            'image_path': original_filepath,
            'timestamp': timestamp
        }

        # Persist embeddings to file
        face_server.save_embeddings()

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
    """Endpoint to get top 3 matching faces from the database"""
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400
    
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
    """Process multiple images from Get/In folder with priority"""
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
        
        # Extract priority if provided
        priority = request.json.get('priority', 5)  # Default priority
        
        try:
            # Add batch request to queue with priority
            face_server.batch_request_queue.put_nowait((priority, (request_id, files_to_process)))
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
    print(f"Starting Face Recognition Server with {face_server.num_worker_threads} workers...")
    print(f"Loaded {len(face_server.face_db)} face embeddings")
    
    # Use eventlet's WSGI server with more workers for better performance
    import eventlet.wsgi
    eventlet.wsgi.server(
        eventlet.listen(('0.0.0.0', 3005)), 
        app,
        max_size=1000,  # Maximum number of concurrent connections
        log_output=False
    )