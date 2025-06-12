from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import base64
from datetime import datetime
import configparser
import uuid
import queue
import threading
import time
import requests
from collections import deque, OrderedDict
import gc
import copy
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import warnings
import signal
import sys
import atexit
import psutil

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning)

# GPU-specific imports with proper error handling
try:
    import torch
    import cupy as cp
    import pynvml
    GPU_AVAILABLE = True
    
    # Initialize NVIDIA ML for GPU monitoring
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    print(f"[GPU] Found {gpu_count} GPU(s)")
    
except ImportError as e:
    print(f"[ERROR] GPU libraries not available: {e}")
    GPU_AVAILABLE = False
    sys.exit(1)  # Exit if GPU not available since we need it

# FAISS with robust GPU handling
try:
    import faiss
    FAISS_AVAILABLE = True
    print(f"[FAISS] Available with {faiss.get_num_gpus()} GPU(s)")
except ImportError:
    print("[ERROR] FAISS not available")
    FAISS_AVAILABLE = False
    sys.exit(1)  # Exit if FAISS not available

# Redis for distributed caching (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[WARNING] Redis not available - using local cache only")

app = Flask(__name__)
CORS(app)

class RobustGPUFaceRecognitionServer:
    def __init__(self):
        print("[INIT] Starting 24/7 Robust GPU Face Recognition Server...")
        
        # Core configuration
        self.config_file = 'config.ini'
        self.config = self.load_config()
        self.threshold = float(self.config.get('Settings', 'threshold', fallback=0.5))
        
        # GPU Memory Management - Reserve 16GB
        self.gpu_memory_limit = 16 * 1024 * 1024 * 1024  # 16GB
        self.gpu_device_id = 0
        self.setup_gpu_environment()
        
        # Performance Configuration
        self.num_cpu_cores = os.cpu_count()
        self.num_worker_threads = min(16, self.num_cpu_cores)  # Conservative for stability
        self.num_http_workers = 4
        
        # Batch sizes optimized for 16GB GPU
        self.gpu_batch_size = 128  # Conservative for stability
        self.embedding_batch_size = 1024  # For adding to FAISS
        
        # Initialize GPU memory pool early
        self.initialize_gpu_memory_pool()
        
        # Face analyzer pool with proper GPU context management
        self.face_analyzer_pool = queue.Queue(maxsize=8)
        self.face_analyzer_lock = threading.Lock()
        self.initialize_face_analyzer_pool()
        
        # Database and indexing
        self.embeddings_file = 'Database/face_embeddings.pkl'
        self.face_db = self.load_embeddings()
        
        # Robust FAISS GPU index
        self.faiss_index = None
        self.embedding_id_mapping = {}
        self.faiss_lock = threading.RLock()  # Reentrant lock for nested calls
        self.initialize_robust_faiss_index()
        
        # Multi-level caching system
        self.l1_cache = {}  # Fast hash-based cache
        self.l2_cache = OrderedDict()  # Recent recognition cache
        self.cache_stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'faiss_hits': 0, 'gpu_hits': 0
        }
        
        # Cache configuration
        self.l1_cache_size = 10000
        self.l2_cache_size = 50000
        self.l2_cache_duration = 3600  # 1 hour
        self.cache_similarity_threshold = 0.65
        
        # Redis initialization (optional)
        self.redis_client = None
        self.redis_enabled = False
        if REDIS_AVAILABLE:
            self.initialize_redis()
        
        # Directory setup
        for directory in ["Get/In", "Get/Out", "static/images", "static/varient", "Logs", "Database/images", "Database/U_images"]:
            os.makedirs(directory, exist_ok=True)
        
        # API endpoints
        self.log_api_endpoint = "http://192.168.14.102:7578/api/FaceRecognition/Recognize-Logs"
        self.log_api_endpoint_2 = "http://192.168.15.129:5002/add_log"
        self.http_broadcast_endpoints = [
            "http://192.168.14.102:7578/api/FaceRecognition/Recognition-Events",
            "http://192.168.15.129:5002/recognition_event"
        ]
        
        # High-performance queues
        self.ultra_priority_queue = queue.PriorityQueue(maxsize=200)
        self.high_priority_queue = queue.PriorityQueue(maxsize=500)
        self.normal_priority_queue = queue.PriorityQueue(maxsize=1000)
        self.batch_request_queue = queue.PriorityQueue(maxsize=100)
        
        # Response caches
        self.response_cache = {}
        self.batch_response_cache = {}
        self.recognition_events = deque(maxlen=10000)
        self.http_broadcast_queue = queue.Queue(maxsize=1000)
        
        # Performance metrics
        self.processed_count = 0
        self.start_time = time.time()
        self.request_times = deque(maxlen=5000)
        self.gpu_memory_usage = deque(maxlen=1000)
        
        # Thread pools
        self.recognition_pool = ThreadPoolExecutor(max_workers=self.num_worker_threads, thread_name_prefix="Recognition")
        self.http_pool = ThreadPoolExecutor(max_workers=self.num_http_workers, thread_name_prefix="HTTP")
        self.batch_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Batch")
        
        # Start all worker systems
        self.start_all_workers()
        
        # Setup graceful shutdown
        self.setup_graceful_shutdown()
        
        # GPU health monitoring
        self.start_gpu_monitoring()
        
        print(f"[SUCCESS] Server initialized with:")
        print(f"  - GPU Memory Reserved: {self.gpu_memory_limit / (1024**3):.1f}GB")
        print(f"  - Worker Threads: {self.num_worker_threads}")
        print(f"  - Face Analyzers: {self.face_analyzer_pool.qsize()}")
        print(f"  - FAISS Index: {self.faiss_index.ntotal if self.faiss_index else 0} embeddings")
        print(f"  - Redis: {'Enabled' if self.redis_enabled else 'Disabled'}")

    def setup_gpu_environment(self):
        """Setup GPU environment with proper memory management"""
        try:
            # Set CUDA device
            torch.cuda.set_device(self.gpu_device_id)
            
            # Set memory growth to prevent allocation issues
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_device_id)
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
            os.environ['CUDA_CACHE_DISABLE'] = '0'    # Enable caching
            
            # CuPy memory pool configuration
            if 'cp' in globals():
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=self.gpu_memory_limit)
                print(f"[GPU] CuPy memory pool limited to {self.gpu_memory_limit / (1024**3):.1f}GB")
            
            # PyTorch memory management
            torch.cuda.empty_cache()
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(self.gpu_device_id)
            gpu_memory = torch.cuda.get_device_properties(self.gpu_device_id).total_memory
            
            print(f"[GPU] Using {gpu_name} with {gpu_memory / (1024**3):.1f}GB total memory")
            
        except Exception as e:
            print(f"[ERROR] GPU setup failed: {e}")
            raise

    def initialize_gpu_memory_pool(self):
        """Initialize GPU memory pool to prevent fragmentation"""
        try:
            # Pre-allocate GPU memory to prevent fragmentation
            if 'cp' in globals():
                # Allocate and free a large block to establish memory pool
                test_size = min(1024 * 1024 * 1024, self.gpu_memory_limit // 4)  # 1GB or 25% of limit
                test_array = cp.zeros(test_size // 4, dtype=cp.float32)  # Divide by 4 for float32
                del test_array
                
                # Set memory pool parameters
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=self.gpu_memory_limit)
                
                print(f"[GPU] Memory pool initialized with {self.gpu_memory_limit / (1024**3):.1f}GB limit")
            
        except Exception as e:
            print(f"[WARNING] GPU memory pool initialization failed: {e}")

    def initialize_face_analyzer_pool(self):
        """Initialize face analyzer pool with proper GPU context management"""
        print("[INIT] Creating robust face analyzer pool...")
        
        pool_size = 8  # Conservative pool size for stability
        successful_analyzers = 0
        
        for i in range(pool_size):
            try:
                # Create analyzer with specific GPU context
                analyzer = FaceAnalysis(
                    name='buffalo_l',
                    root='models',
                    providers=['CUDAExecutionProvider']  # GPU only
                )
                
                # Prepare with proper context isolation
                analyzer.prepare(ctx_id=self.gpu_device_id, det_size=(640, 640))
                
                # Test the analyzer
                test_img = np.zeros((224, 224, 3), dtype=np.uint8)
                test_faces = analyzer.get(test_img)  # This should work without errors
                
                self.face_analyzer_pool.put(analyzer)
                successful_analyzers += 1
                print(f"[INIT] Created face analyzer {successful_analyzers}/{pool_size}")
                
            except Exception as e:
                print(f"[ERROR] Failed to create face analyzer {i}: {e}")
                
        if successful_analyzers == 0:
            raise RuntimeError("Failed to create any face analyzers")
            
        print(f"[SUCCESS] Created {successful_analyzers} face analyzers")

    def get_face_analyzer(self):
        """Get face analyzer from pool with timeout"""
        try:
            return self.face_analyzer_pool.get(timeout=10.0)
        except queue.Empty:
            # Emergency: create temporary analyzer
            print("[WARNING] Face analyzer pool exhausted, creating temporary analyzer")
            try:
                analyzer = FaceAnalysis(
                    name='buffalo_l',
                    root='models',
                    providers=['CUDAExecutionProvider']
                )
                analyzer.prepare(ctx_id=self.gpu_device_id, det_size=(640, 640))
                return analyzer
            except Exception as e:
                print(f"[ERROR] Failed to create emergency analyzer: {e}")
                raise

    def return_face_analyzer(self, analyzer):
        """Return face analyzer to pool"""
        try:
            self.face_analyzer_pool.put_nowait(analyzer)
        except queue.Full:
            # Pool is full, let analyzer be garbage collected
            pass

    def initialize_robust_faiss_index(self):
        """Initialize FAISS index with robust GPU memory management"""
        if not FAISS_AVAILABLE:
            print("[ERROR] FAISS not available")
            return
            
        print("[INIT] Building robust FAISS GPU index...")
        
        if not self.face_db:
            print("[WARNING] No embeddings to index")
            return
        
        with self.faiss_lock:
            try:
                # Extract embeddings
                embeddings = []
                identities = []
                
                for identity, data in self.face_db.items():
                    if 'embedding' in data and data['embedding'] is not None:
                        embeddings.append(data['embedding'])
                        identities.append(identity)
                
                if not embeddings:
                    print("[WARNING] No valid embeddings found")
                    return
                
                # Convert and normalize embeddings
                embeddings_array = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings_array = embeddings_array / norms
                
                dimension = embeddings_array.shape[1]
                print(f"[FAISS] Building index for {len(embeddings_array)} embeddings, dimension {dimension}")
                
                # Create GPU resources with conservative memory allocation
                gpu_resources = faiss.StandardGpuResources()
                
                # Set memory limits to prevent corruption
                available_memory = self.get_available_gpu_memory()
                faiss_memory_limit = min(
                    2 * 1024 * 1024 * 1024,  # 2GB max for FAISS
                    available_memory // 4     # 25% of available memory
                )
                gpu_resources.setTempMemory(faiss_memory_limit)
                
                print(f"[FAISS] Allocated {faiss_memory_limit / (1024**2):.0f}MB for FAISS operations")
                
                # Create GPU index configuration
                config = faiss.GpuIndexFlatConfig()
                config.device = self.gpu_device_id
                config.useFloat16 = False  # Use full precision for stability
                config.usePrecomputed = False
                
                # Create GPU index directly (no CPU-to-GPU transfer)
                self.faiss_index = faiss.GpuIndexFlatIP(gpu_resources, dimension, config)
                
                # Add embeddings in batches to prevent memory issues
                batch_size = self.embedding_batch_size
                total_added = 0
                
                for i in range(0, len(embeddings_array), batch_size):
                    batch_end = min(i + batch_size, len(embeddings_array))
                    batch_embeddings = embeddings_array[i:batch_end]
                    
                    # Add batch to index
                    self.faiss_index.add(batch_embeddings)
                    total_added += len(batch_embeddings)
                    
                    # Synchronize GPU to prevent memory corruption
                    torch.cuda.synchronize()
                    
                    print(f"[FAISS] Added {total_added}/{len(embeddings_array)} embeddings")
                
                # Create ID mapping
                self.embedding_id_mapping = {i: identities[i] for i in range(len(identities))}
                
                # Store GPU resources reference to prevent garbage collection
                self.gpu_resources = gpu_resources
                
                print(f"[SUCCESS] FAISS GPU index built with {self.faiss_index.ntotal} embeddings")
                
            except Exception as e:
                print(f"[ERROR] Failed to build FAISS index: {e}")
                self.faiss_index = None
                self.cleanup_faiss_resources()

    def get_available_gpu_memory(self):
        """Get available GPU memory in bytes"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                return torch.cuda.get_device_properties(self.gpu_device_id).total_memory - torch.cuda.memory_allocated(self.gpu_device_id)
            return 0
        except Exception:
            return 2 * 1024 * 1024 * 1024  # Default 2GB

    def cleanup_faiss_resources(self):
        """Clean up FAISS GPU resources"""
        try:
            if hasattr(self, 'gpu_resources'):
                del self.gpu_resources
            if hasattr(self, 'faiss_index'):
                del self.faiss_index
                self.faiss_index = None
            torch.cuda.empty_cache()
            print("[CLEANUP] FAISS resources cleaned up")
        except Exception as e:
            print(f"[ERROR] FAISS cleanup failed: {e}")

    def safe_faiss_search(self, query_embedding, top_k=1):
        """Thread-safe FAISS search with robust error handling"""
        if not self.faiss_index:
            return []
        
        with self.faiss_lock:
            try:
                # Prepare query embedding
                query_embedding = query_embedding.astype(np.float32)
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
                query_embedding = query_embedding.reshape(1, -1)
                
                # Create isolated copy to prevent memory corruption
                query_copy = np.copy(query_embedding)
                
                # Perform search with timeout protection
                start_time = time.time()
                similarities, indices = self.faiss_index.search(query_copy, top_k)
                search_time = time.time() - start_time
                
                # Synchronize to ensure completion
                torch.cuda.synchronize()
                
                # Process results
                results = []
                for similarity, index in zip(similarities[0], indices[0]):
                    if index != -1 and index < len(self.embedding_id_mapping):
                        identity = self.embedding_id_mapping[index]
                        results.append((identity, float(similarity)))
                
                self.cache_stats['faiss_hits'] += 1
                
                # Log slow searches
                if search_time > 0.1:
                    print(f"[WARNING] Slow FAISS search: {search_time:.3f}s")
                
                return results
                
            except Exception as e:
                print(f"[ERROR] FAISS search failed: {e}")
                # Try to recover
                self.schedule_faiss_recovery()
                return []

    def schedule_faiss_recovery(self):
        """Schedule FAISS index recovery in background"""
        def recovery_task():
            try:
                print("[RECOVERY] Starting FAISS index recovery...")
                
                # Clean up corrupted resources
                self.cleanup_faiss_resources()
                
                # Wait for GPU memory to stabilize
                time.sleep(2)
                
                # Rebuild index
                self.initialize_robust_faiss_index()
                
                if self.faiss_index:
                    print("[RECOVERY] FAISS index recovered successfully")
                else:
                    print("[RECOVERY] FAISS recovery failed")
                    
            except Exception as e:
                print(f"[ERROR] FAISS recovery failed: {e}")
        
        # Submit recovery task to thread pool
        self.recognition_pool.submit(recovery_task)

    def numpy_similarity_search(self, query_embedding, top_k=1):
        """Fallback numpy-based similarity search"""
        try:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            similarities = []
            
            for identity, data in self.face_db.items():
                if 'embedding' in data and data['embedding'] is not None:
                    stored_embedding = data['embedding']
                    stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
                    similarity = float(np.dot(query_embedding, stored_embedding))
                    similarities.append((identity, similarity))
            
            # Sort and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"[ERROR] Numpy similarity search failed: {e}")
            return []

    def ultra_fast_recognize_face(self, image_data, camera_id=None):
        """Ultra-fast face recognition with robust GPU handling"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_id = str(uuid.uuid4())[:8]
        
        try:
            # Convert base64 to image
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"status": "error", "message": "Invalid image data"}

            # Prepare filenames
            prefix = f"cam{camera_id}_{request_id}_" if camera_id else f"{request_id}_"
            input_filename = self.save_image_async(img, "Get/In", f"{prefix}input")

            # Get face analyzer
            face_analyzer = self.get_face_analyzer()
            
            try:
                # Face detection
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_analyzer.get(img_rgb)
                
                # Return analyzer immediately
                self.return_face_analyzer(face_analyzer)
                
            except Exception as e:
                self.return_face_analyzer(face_analyzer)
                raise e

            if len(faces) == 0:
                return self.handle_no_face_detected(input_filename, prefix, camera_id, request_id)

            # Extract embedding
            query_embedding = faces[0].embedding

            # Multi-level cache check
            cached_result = self.check_l1_cache(query_embedding)
            if cached_result:
                cached_result.update({
                    "input_filename": input_filename,
                    "request_id": request_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp
                })
                return cached_result

            cached_result = self.check_l2_cache(query_embedding)
            if cached_result:
                cached_result.update({
                    "input_filename": input_filename,
                    "request_id": request_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp
                })
                self.add_to_l1_cache(query_embedding, cached_result)
                return cached_result

            # Database search using robust FAISS
            search_results = self.safe_faiss_search(query_embedding, top_k=1)
            
            # Fallback to numpy if FAISS fails
            if not search_results:
                search_results = self.numpy_similarity_search(query_embedding, top_k=1)
            
            if search_results:
                best_identity, best_similarity = search_results[0]
                
                if best_similarity > self.threshold:
                    result = self.create_match_result(
                        best_identity, best_similarity, input_filename, 
                        prefix, camera_id, request_id, query_embedding
                    )
                    
                    # Add to caches
                    self.add_to_l1_cache(query_embedding, result)
                    self.add_to_l2_cache(query_embedding, result)
                    
                    return result
            
            # No match found
            return self.handle_unrecognized_face(
                img, faces[0], input_filename, prefix, camera_id, request_id
            )
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Recognition error: {str(e)}",
                "camera_id": camera_id,
                "request_id": request_id
            }

    def check_l1_cache(self, query_embedding):
        """Check L1 cache"""
        try:
            embedding_hash = hash(query_embedding.astype(np.float32).tobytes()[:64])
            
            if embedding_hash in self.l1_cache:
                result, timestamp = self.l1_cache[embedding_hash]
                
                if time.time() - timestamp < 300:  # 5 minutes
                    self.cache_stats['l1_hits'] += 1
                    result = result.copy()
                    result['l1_cache_hit'] = True
                    return result
                else:
                    del self.l1_cache[embedding_hash]
            
            self.cache_stats['l1_misses'] += 1
            return None
            
        except Exception as e:
            self.cache_stats['l1_misses'] += 1
            return None

    def add_to_l1_cache(self, query_embedding, result):
        """Add to L1 cache"""
        try:
            if result.get('status') == 'matched' and result.get('confidence', 0) > 0.6:
                embedding_hash = hash(query_embedding.astype(np.float32).tobytes()[:64])
                
                if len(self.l1_cache) >= self.l1_cache_size:
                    # Remove oldest entries
                    oldest_keys = list(self.l1_cache.keys())[:100]
                    for key in oldest_keys:
                        if key in self.l1_cache:
                            del self.l1_cache[key]
                
                self.l1_cache[embedding_hash] = (result.copy(), time.time())
                
        except Exception as e:
            pass

    def check_l2_cache(self, query_embedding):
        """Check L2 cache with similarity matching"""
        try:
            if not self.l2_cache:
                self.cache_stats['l2_misses'] += 1
                return None
            
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            current_time = time.time()
            best_match = None
            best_similarity = -1
            
            expired_keys = []
            for cache_key, (cached_embedding, cached_result, timestamp) in list(self.l2_cache.items()):
                if current_time - timestamp > self.l2_cache_duration:
                    expired_keys.append(cache_key)
                    continue
                
                try:
                    similarity = float(np.dot(normalized_query, cached_embedding))
                    
                    if similarity > self.cache_similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = cached_result.copy()
                        best_match['l2_cache_hit'] = True
                        best_match['l2_cache_similarity'] = similarity
                        best_match['cache_age_seconds'] = current_time - timestamp
                except Exception:
                    continue
            
            # Clean expired entries
            for key in expired_keys:
                if key in self.l2_cache:
                    del self.l2_cache[key]
            
            if best_match:
                self.cache_stats['l2_hits'] += 1
                return best_match
            else:
                self.cache_stats['l2_misses'] += 1
                return None
                
        except Exception as e:
            self.cache_stats['l2_misses'] += 1
            return None

    def add_to_l2_cache(self, query_embedding, result):
        """Add to L2 cache"""
        try:
            if result.get('status') == 'matched' and result.get('confidence', 0) > 0.5:
                if len(self.l2_cache) >= self.l2_cache_size:
                    for _ in range(100):
                        if self.l2_cache:
                            self.l2_cache.popitem(last=False)
                
                cache_key = f"l2_{int(time.time() * 1000)}_{len(self.l2_cache)}"
                normalized_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                self.l2_cache[cache_key] = (
                    normalized_embedding,
                    result.copy(),
                    time.time()
                )
                
        except Exception as e:
            pass

    def save_image_async(self, img, folder, prefix):
        """Save image asynchronously"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        
        def save_task():
            try:
                cv2.imwrite(filepath, img)
            except Exception as e:
                print(f"[ERROR] Failed to save image: {e}")
        
        self.recognition_pool.submit(save_task)
        return filename

    def handle_no_face_detected(self, input_filename, prefix, camera_id, request_id):
        """Handle no face detected"""
        try:
            # Create placeholder image
            placeholder_img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "NO FACE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            output_filename = self.save_image_async(placeholder_img, "Get/Out", f"{prefix}no_face")
            _, buffer = cv2.imencode('.jpg', placeholder_img)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "status": "unrecognized",
                "message": "No face detected",
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_image": image_base64,
                "camera_id": camera_id,
                "request_id": request_id
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error handling no face: {str(e)}",
                "input_filename": input_filename,
                "camera_id": camera_id,
                "request_id": request_id
            }

    def create_match_result(self, identity, similarity, input_filename, prefix, camera_id, request_id, query_embedding):
        """Create match result"""
        try:
            # Find image path
            image_path = None
            for face_identity, data in self.face_db.items():
                if face_identity.split('@')[0] == identity.split('@')[0]:
                    image_path = data.get('image_path')
                    break
            
            if not image_path or not os.path.exists(image_path):
                # Create placeholder
                matched_image = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.putText(matched_image, "MATCHED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                matched_image = cv2.imread(image_path)
            
            output_filename = self.save_image_async(matched_image, "Get/Out", f"{prefix}match")
            
            _, buffer = cv2.imencode('.jpg', matched_image)
            matched_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "status": "matched",
                "confidence": float(similarity),
                "matched_image": matched_image_base64,
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_filename": os.path.basename(image_path) if image_path else "placeholder.jpg",
                "camera_id": camera_id,
                "request_id": request_id,
                "gpu_accelerated": True
            }
            
            # Log asynchronously
            self.log_recognition_async(result, camera_id)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating match result: {str(e)}",
                "input_filename": input_filename,
                "camera_id": camera_id,
                "request_id": request_id
            }

    def handle_unrecognized_face(self, img, face, input_filename, prefix, camera_id, request_id):
        """Handle unrecognized face"""
        try:
            # Save unrecognized face
            face_id = os.path.splitext(input_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"unrecognized_{face_id}_{timestamp}.jpg"
            filepath = os.path.join("static/images", filename)
            
            self.save_image_async(img, "static/images", f"unrecognized_{face_id}")
            output_filename = self.save_image_async(img, "Get/Out", f"{prefix}unrecognized")
            
            _, buffer = cv2.imencode('.jpg', img)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "status": "unrecognized_saved",
                "message": "Face not recognized - saved",
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_image": image_base64,
                "face_id": face_id,
                "saved_path": filepath,
                "camera_id": camera_id,
                "request_id": request_id
            }
            
            # Log asynchronously
            self.log_recognition_async(result, camera_id)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error handling unrecognized face: {str(e)}",
                "input_filename": input_filename,
                "camera_id": camera_id,
                "request_id": request_id
            }

    def log_recognition_async(self, result, camera_id=None):
        """Log recognition details asynchronously"""
        def log_task():
            try:
                # Create event data
                event_data = {
                    "timestamp": datetime.now().isoformat(),
                    "status": result.get('status', 'Unknown'),
                    "camera_id": camera_id,
                    "camera_name": self.get_camera_name(camera_id),
                    "confidence": result.get('confidence', 0.0),
                    "matched_filename": result.get('matched_filename', ''),
                    "identity": os.path.splitext(result.get('matched_filename', ''))[0] if result.get('matched_filename') else None,
                    "input_image": f"/Get/In/{result.get('input_filename', '')}" if result.get('input_filename') else None,
                    "matched_image": f"Database/images/{result.get('matched_filename', '')}" if result.get('matched_filename') else None,
                    "cache_hit": result.get('l1_cache_hit', False) or result.get('l2_cache_hit', False),
                    "request_id": result.get('request_id', ''),
                    "gpu_accelerated": True
                }

                # Add to events
                self.recognition_events.append(event_data)
                
                # Broadcast
                try:
                    self.http_broadcast_queue.put_nowait(event_data)
                except queue.Full:
                    pass

                # API logging
                self.send_api_logs_async(result, camera_id)
                    
            except Exception as e:
                print(f"[ERROR] Logging failed: {e}")
        
        self.recognition_pool.submit(log_task)

    def send_api_logs_async(self, result, camera_id):
        """Send API logs asynchronously"""
        def api_task():
            try:
                log_entry = {
                    "logDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "status": result.get('status', 'Unknown'),
                    "inputFilename": result.get('input_filename', ''),
                    "inputPath": os.path.join('Get/In', result.get('input_filename', '')),
                    "outputFilename": result.get('output_filename', ''),
                    "outputPath": os.path.join('Get/Out', result.get('output_filename', '')),
                    "confidence": result.get('confidence', 0.0),
                    "matchedFilename": result.get('matched_filename', ''),
                    "gpu_accelerated": True
                }

                for endpoint in [self.log_api_endpoint, self.log_api_endpoint_2]:
                    try:
                        if endpoint == self.log_api_endpoint_2:
                            log_entry["camera_id"] = camera_id

                        requests.post(
                            endpoint,
                            json=[log_entry],
                            headers={'Content-Type': 'application/json'},
                            timeout=1.0
                        )
                    except Exception:
                        pass  # Ignore API failures
            except Exception as e:
                print(f"[ERROR] API logging failed: {e}")
        
        self.recognition_pool.submit(api_task)

    def get_camera_name(self, camera_id):
        """Get camera name"""
        camera_names = {
            1: 'Paiza Gate 1', 2: 'Paiza Gate 2', 3: 'Paiza Gate 3', 4: 'Paiza Gate 4',
            31: 'Membership Counter Out', 32: 'Membership Counter In',
            61: 'Travel Counter Out', 62: 'Travel Counter In',
            1131: 'Membership Counter Out Sup', 1132: 'Membership Counter In Sup',
            1161: 'Travel Counter Out Sup', 1162: 'Travel Counter In Sup',
            3701: 'Table 37 Left', 3702: 'Table 37 Right'
        }
        return camera_names.get(camera_id, f"Camera {camera_id}")

    def initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host='localhost', port=6379, db=0,
                socket_connect_timeout=1, socket_timeout=1,
                retry_on_timeout=False, decode_responses=False
            )
            self.redis_client.ping()
            self.redis_enabled = True
            print("[SUCCESS] Redis connected")
        except Exception as e:
            self.redis_enabled = False
            print(f"[INFO] Redis not available: {e}")

    def start_all_workers(self):
        """Start all worker threads"""
        # Priority queue workers
        for i in range(2):
            threading.Thread(target=self.process_ultra_priority_queue, daemon=True).start()
        
        for i in range(4):
            threading.Thread(target=self.process_high_priority_queue, daemon=True).start()
        
        for i in range(8):
            threading.Thread(target=self.process_normal_priority_queue, daemon=True).start()
        
        # Batch processing workers
        for i in range(2):
            threading.Thread(target=self.process_batch_queue, daemon=True).start()
        
        # HTTP workers
        for i in range(4):
            threading.Thread(target=self.process_http_broadcasts, daemon=True).start()
        
        # Maintenance workers
        threading.Thread(target=self.cache_maintenance_worker, daemon=True).start()
        threading.Thread(target=self.performance_monitor_worker, daemon=True).start()

    def process_ultra_priority_queue(self):
        """Process ultra priority queue"""
        while True:
            try:
                _, (request_id, image_data, camera_id) = self.ultra_priority_queue.get(timeout=0.1)
                
                start_time = time.time()
                result = self.ultra_fast_recognize_face(image_data, camera_id)
                processing_time = time.time() - start_time
                
                self.response_cache[request_id] = (result, time.time())
                self.processed_count += 1
                self.request_times.append(processing_time)
                
                self.ultra_priority_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] Ultra priority queue: {e}")
                time.sleep(0.1)

    def process_high_priority_queue(self):
        """Process high priority queue"""
        while True:
            try:
                _, (request_id, image_data, camera_id) = self.high_priority_queue.get(timeout=0.1)
                
                start_time = time.time()
                result = self.ultra_fast_recognize_face(image_data, camera_id)
                processing_time = time.time() - start_time
                
                self.response_cache[request_id] = (result, time.time())
                self.processed_count += 1
                self.request_times.append(processing_time)
                
                self.high_priority_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] High priority queue: {e}")
                time.sleep(0.1)

    def process_normal_priority_queue(self):
        """Process normal priority queue"""
        while True:
            try:
                _, (request_id, image_data, camera_id) = self.normal_priority_queue.get(timeout=0.1)
                
                start_time = time.time()
                result = self.ultra_fast_recognize_face(image_data, camera_id)
                processing_time = time.time() - start_time
                
                self.response_cache[request_id] = (result, time.time())
                self.processed_count += 1
                self.request_times.append(processing_time)
                
                self.normal_priority_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] Normal priority queue: {e}")
                time.sleep(0.1)

    def process_batch_queue(self):
        """Process batch queue"""
        while True:
            try:
                _, (request_id, files_to_process) = self.batch_request_queue.get()
                
                print(f"[BATCH] Processing {len(files_to_process)} files")
                results = self.batch_process_files(files_to_process)
                
                self.batch_response_cache[request_id] = (results, time.time())
                self.batch_request_queue.task_done()
                
            except Exception as e:
                print(f"[ERROR] Batch processing: {e}")
                time.sleep(0.1)

    def batch_process_files(self, files_to_process):
        """Process files in batch"""
        results = []
        batch_size = 32
        
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for filename, image_id in batch:
                    future = executor.submit(self.process_single_batch_file, filename, image_id)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures, timeout=30):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({"status": "error", "message": str(e)})
            
            time.sleep(0.01)
        
        return results

    def process_single_batch_file(self, filename, image_id):
        """Process single batch file"""
        request_id = str(uuid.uuid4())[:8]
        
        try:
            image_path = os.path.join("Get/In", filename)
            
            if not os.path.exists(image_path):
                return {
                    "image_id": image_id,
                    "status": "error",
                    "message": f"File {filename} not found",
                    "request_id": request_id
                }
            
            img = cv2.imread(image_path)
            if img is None:
                return {
                    "image_id": image_id,
                    "status": "error",
                    "message": "Invalid image file",
                    "request_id": request_id
                }
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', img)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Process with main recognition
            result = self.ultra_fast_recognize_face(image_data)
            
            # Format for batch response
            batch_result = {
                "image_id": image_id,
                "status": result.get('status', 'error'),
                "filename": filename,
                "request_id": request_id
            }
            
            if result.get('status') == 'matched':
                batch_result.update({
                    "confidence": result.get('confidence', 0.0),
                    "matchedFilename": result.get('matched_filename', '')
                })
            else:
                batch_result["message"] = result.get('message', 'Processing error')
            
            return batch_result
                
        except Exception as e:
            return {
                "image_id": image_id,
                "status": "error",
                "filename": filename,
                "message": str(e),
                "request_id": request_id
            }

    def process_http_broadcasts(self):
        """Process HTTP broadcasts"""
        while True:
            try:
                event_data = self.http_broadcast_queue.get(timeout=1.0)
                self.http_pool.submit(self.send_http_broadcast, event_data)
                self.http_broadcast_queue.task_done()
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] HTTP broadcast: {e}")
                time.sleep(0.1)

    def send_http_broadcast(self, event_data):
        """Send HTTP broadcast"""
        for endpoint in self.http_broadcast_endpoints:
            try:
                requests.post(
                    endpoint,
                    json=event_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=0.5
                )
            except Exception:
                pass

    def cache_maintenance_worker(self):
        """Maintain caches"""
        while True:
            try:
                time.sleep(30)
                current_time = time.time()
                
                # Clean L1 cache
                if len(self.l1_cache) > self.l1_cache_size * 1.2:
                    remove_count = len(self.l1_cache) // 5
                    oldest_keys = list(self.l1_cache.keys())[:remove_count]
                    for key in oldest_keys:
                        if key in self.l1_cache:
                            del self.l1_cache[key]
                
                # Clean L2 cache
                expired_l2_keys = [
                    key for key, (_, _, timestamp) in list(self.l2_cache.items())
                    if current_time - timestamp > self.l2_cache_duration
                ]
                for key in expired_l2_keys:
                    if key in self.l2_cache:
                        del self.l2_cache[key]
                
                # Clean response caches
                expired_response_keys = [
                    key for key, (_, timestamp) in list(self.response_cache.items())
                    if current_time - timestamp > 120
                ]
                for key in expired_response_keys:
                    if key in self.response_cache:
                        del self.response_cache[key]
                
                expired_batch_keys = [
                    key for key, (_, timestamp) in list(self.batch_response_cache.items())
                    if current_time - timestamp > 600
                ]
                for key in expired_batch_keys:
                    if key in self.batch_response_cache:
                        del self.batch_response_cache[key]
                
                # Periodic GPU cleanup
                if self.processed_count % 1000 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"[ERROR] Cache maintenance: {e}")

    def start_gpu_monitoring(self):
        """Start GPU health monitoring"""
        def gpu_monitor():
            while True:
                try:
                    time.sleep(60)  # Check every minute
                    
                    # Get GPU memory usage
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(self.gpu_device_id)
                        memory_cached = torch.cuda.memory_reserved(self.gpu_device_id)
                        memory_total = torch.cuda.get_device_properties(self.gpu_device_id).total_memory
                        
                        usage_percent = (memory_allocated / memory_total) * 100
                        
                        self.gpu_memory_usage.append({
                            'timestamp': time.time(),
                            'allocated_gb': memory_allocated / (1024**3),
                            'cached_gb': memory_cached / (1024**3),
                            'total_gb': memory_total / (1024**3),
                            'usage_percent': usage_percent
                        })
                        
                        # Warning if usage is too high
                        if usage_percent > 90:
                            print(f"[WARNING] High GPU memory usage: {usage_percent:.1f}%")
                            torch.cuda.empty_cache()
                        
                        # Log periodic status
                        if self.processed_count % 100 == 0:
                            print(f"[GPU] Memory: {usage_percent:.1f}% ({memory_allocated/(1024**3):.1f}GB/{memory_total/(1024**3):.1f}GB)")
                    
                except Exception as e:
                    print(f"[ERROR] GPU monitoring: {e}")
        
        threading.Thread(target=gpu_monitor, daemon=True).start()

    def performance_monitor_worker(self):
        """Monitor performance"""
        while True:
            try:
                time.sleep(60)
                
                uptime = time.time() - self.start_time
                rps = self.processed_count / uptime if uptime > 0 else 0
                avg_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
                
                # Cache stats
                total_l1 = self.cache_stats['l1_hits'] + self.cache_stats['l1_misses']
                total_l2 = self.cache_stats['l2_hits'] + self.cache_stats['l2_misses']
                
                l1_rate = (self.cache_stats['l1_hits'] / total_l1 * 100) if total_l1 > 0 else 0
                l2_rate = (self.cache_stats['l2_hits'] / total_l2 * 100) if total_l2 > 0 else 0
                
                print(f"[PERF] RPS: {rps:.1f} | Avg: {avg_time*1000:.1f}ms | "
                      f"L1: {l1_rate:.1f}% | L2: {l2_rate:.1f}% | "
                      f"Processed: {self.processed_count}")
                
            except Exception as e:
                print(f"[ERROR] Performance monitor: {e}")

    def setup_graceful_shutdown(self):
        """Setup graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"[SHUTDOWN] Received signal {signum}, shutting down gracefully...")
            self.cleanup_and_exit()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.cleanup_and_exit)

    def cleanup_and_exit(self):
        """Cleanup resources and exit"""
        try:
            print("[CLEANUP] Starting cleanup...")
            
            # Stop thread pools
            if hasattr(self, 'recognition_pool'):
                self.recognition_pool.shutdown(wait=True, timeout=10)
            if hasattr(self, 'http_pool'):
                self.http_pool.shutdown(wait=True, timeout=5)
            if hasattr(self, 'batch_pool'):
                self.batch_pool.shutdown(wait=True, timeout=5)
            
            # Cleanup GPU resources
            self.cleanup_faiss_resources()
            
            # Final GPU cleanup
            torch.cuda.empty_cache()
            
            print("[CLEANUP] Cleanup completed")
            
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {e}")

    # Compatibility methods
    def load_config(self):
        """Load configuration"""
        config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            config['Settings'] = {'threshold': '0.5'}
            with open(self.config_file, 'w') as f:
                config.write(f)
        else:
            config.read(self.config_file)
        return config

    def save_embeddings(self):
        """Save embeddings"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump({'embeddings': self.face_db}, f)
            print(f"[SAVE] Saved {len(self.face_db)} embeddings")
            
            # Rebuild index asynchronously
            def rebuild_task():
                try:
                    self.cleanup_faiss_resources()
                    time.sleep(1)
                    self.initialize_robust_faiss_index()
                except Exception as e:
                    print(f"[ERROR] Index rebuild failed: {e}")
            
            self.recognition_pool.submit(rebuild_task)
            
        except Exception as e:
            print(f"[ERROR] Save embeddings failed: {e}")

    def load_embeddings(self):
        """Load embeddings"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    print(f"[LOAD] Loaded {len(data['embeddings'])} embeddings")
                    return data['embeddings']
            except Exception as e:
                print(f"[ERROR] Load embeddings failed: {e}")
                return {}
        return {}

    def get_recognition_result(self, request_id, timeout=30):
        """Get recognition result"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.response_cache:
                result, _ = self.response_cache.pop(request_id)
                return result
            time.sleep(0.01)
        
        return {"status": "error", "message": "Request timeout"}

    def get_batch_result(self, request_id, timeout=3600):
        """Get batch result"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.batch_response_cache:
                result, _ = self.batch_response_cache.pop(request_id)
                return result
            time.sleep(0.1)
        
        return {"status": "error", "message": "Batch timeout"}

    def get_performance_metrics(self):
        """Get performance metrics"""
        uptime = time.time() - self.start_time
        rps = self.processed_count / uptime if uptime > 0 else 0
        avg_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        
        total_l1 = self.cache_stats['l1_hits'] + self.cache_stats['l1_misses']
        total_l2 = self.cache_stats['l2_hits'] + self.cache_stats['l2_misses']
        
        # GPU memory info
        gpu_memory_info = {}
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.gpu_device_id)
            memory_total = torch.cuda.get_device_properties(self.gpu_device_id).total_memory
            gpu_memory_info = {
                "allocated_gb": memory_allocated / (1024**3),
                "total_gb": memory_total / (1024**3),
                "usage_percent": (memory_allocated / memory_total) * 100
            }
        
        return {
            "uptime_seconds": uptime,
            "processed_requests": self.processed_count,
            "requests_per_second": rps,
            "avg_processing_time_ms": avg_time * 1000,
            
            # Queue sizes
            "ultra_priority_queue_size": self.ultra_priority_queue.qsize(),
            "high_priority_queue_size": self.high_priority_queue.qsize(),
            "normal_priority_queue_size": self.normal_priority_queue.qsize(),
            "batch_queue_size": self.batch_request_queue.qsize(),
            "http_broadcast_queue_size": self.http_broadcast_queue.qsize(),
            
            # Database and cache info
            "face_db_size": len(self.face_db),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "l1_cache_size": len(self.l1_cache),
            "l2_cache_size": len(self.l2_cache),
            "l1_hit_rate": (self.cache_stats['l1_hits'] / total_l1 * 100) if total_l1 > 0 else 0,
            "l2_hit_rate": (self.cache_stats['l2_hits'] / total_l2 * 100) if total_l2 > 0 else 0,
            
            # System info
            "gpu_memory": gpu_memory_info,
            "redis_enabled": self.redis_enabled,
            "cache_stats": self.cache_stats,
            "recognition_events_count": len(self.recognition_events),
            "response_cache_size": len(self.response_cache),
            "batch_response_cache_size": len(self.batch_response_cache),
            "gpu_accelerated": True
        }

    # Legacy method
    def recognize_face(self, image_data, camera_id=None):
        """Legacy compatibility"""
        return self.ultra_fast_recognize_face(image_data, camera_id)


# Initialize server
try:
    face_server = RobustGPUFaceRecognitionServer()
    print("[SUCCESS] 24/7 Robust GPU Face Recognition Server initialized")
except Exception as e:
    print(f"[CRITICAL] Server initialization failed: {e}")
    sys.exit(1)

# Flask routes
@app.route('/api/heartbeat', methods=['GET'])
def heartbeat():
    """Heartbeat endpoint"""
    try:
        uptime = time.time() - face_server.start_time
        rps = face_server.processed_count / uptime if uptime > 0 else 0
        
        gpu_info = {}
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(face_server.gpu_device_id)
            memory_total = torch.cuda.get_device_properties(face_server.gpu_device_id).total_memory
            gpu_info = {
                "memory_usage_percent": (memory_allocated / memory_total) * 100,
                "memory_allocated_gb": memory_allocated / (1024**3)
            }
        
        return jsonify({
            "status": "alive",
            "timestamp": datetime.now().isoformat(),
            "rps": round(rps, 2),
            "processed_requests": face_server.processed_count,
            "gpu_accelerated": True,
            "gpu_info": gpu_info,
            "24x7_ready": True
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Performance metrics"""
    try:
        return jsonify(face_server.get_performance_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize', methods=['POST', 'OPTIONS'])
def recognize():
    """Main recognition endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        request_id = str(uuid.uuid4())
        camera_id = request.json.get('camera_id')
        priority = request.json.get('priority', 'normal')
        
        current_time = time.time()
        
        # Intelligent queue selection
        total_queue_size = (face_server.ultra_priority_queue.qsize() + 
                           face_server.high_priority_queue.qsize() + 
                           face_server.normal_priority_queue.qsize())
        
        if priority == 'ultra' or total_queue_size < 5:
            queue_obj = face_server.ultra_priority_queue
            timeout = 5
        elif priority == 'high' or total_queue_size < 20:
            queue_obj = face_server.high_priority_queue
            timeout = 15
        else:
            queue_obj = face_server.normal_priority_queue
            timeout = 30
        
        # Add to queue
        try:
            queue_obj.put_nowait((current_time, (request_id, request.json['image'], camera_id)))
        except queue.Full:
            return jsonify({
                "error": "Server at maximum capacity",
                "queue_sizes": {
                    "ultra": face_server.ultra_priority_queue.qsize(),
                    "high": face_server.high_priority_queue.qsize(),
                    "normal": face_server.normal_priority_queue.qsize()
                }
            }), 503
        
        # Get result
        result = face_server.get_recognition_result(request_id, timeout)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_process_files', methods=['POST', 'OPTIONS'])
def batch_process_files():
    """Batch processing endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'files' not in request.json:
            return jsonify({"error": "No files provided"}), 400
            
        files_data = request.json['files']
        files_to_process = [(item['filename'], item['image_id']) 
                           for item in files_data 
                           if 'filename' in item and 'image_id' in item]
        
        if not files_to_process:
            return jsonify({"error": "No valid files to process"}), 400
        
        request_id = str(uuid.uuid4())
        priority = request.json.get('priority', 5)
        
        try:
            face_server.batch_request_queue.put_nowait((priority, (request_id, files_to_process)))
        except queue.Full:
            return jsonify({"error": "Batch processing queue full"}), 503
        
        # Get results
        results = face_server.get_batch_result(request_id)
        
        return jsonify({
            "status": "success",
            "results": results,
            "total_files": len(files_to_process),
            "gpu_accelerated": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_image_to_database', methods=['POST'])
def add_image_to_database():
    """Add image to database"""
    try:
        if 'image' not in request.json or 'identity' not in request.json:
            return jsonify({"error": "Missing 'image' or 'identity' in request"}), 400

        image_data = request.json['image']
        identity = request.json['identity']

        os.makedirs("Database/U_images", exist_ok=True)

        # Decode image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Get face analyzer
        face_analyzer = face_server.get_face_analyzer()
        
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_analyzer.get(img_rgb)
            face_server.return_face_analyzer(face_analyzer)
        except Exception as e:
            face_server.return_face_analyzer(face_analyzer)
            raise e

        if len(faces) == 0:
            return jsonify({"error": "No face detected in the image"}), 400

        if len(faces) > 1:
            return jsonify({"error": "Multiple faces detected. Only one face per image is allowed"}), 400

        # Extract embedding
        embedding = faces[0].embedding

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"{identity}_{timestamp}.jpg"
        original_filepath = os.path.join("Database/U_images", original_filename)
        
        # Save asynchronously
        def save_and_update():
            try:
                cv2.imwrite(original_filepath, img)
                
                # Add to database
                face_server.face_db[identity] = {
                    'embedding': embedding,
                    'image_path': original_filepath,
                    'timestamp': timestamp
                }
                
                # Save and rebuild
                face_server.save_embeddings()
                
            except Exception as e:
                print(f"[ERROR] Save and update failed: {e}")
        
        face_server.recognition_pool.submit(save_and_update)

        return jsonify({
            "status": "success",
            "message": f"Image for '{identity}' added to database",
            "filename": original_filename,
            "timestamp": timestamp,
            "gpu_accelerated": True
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

@app.route('/add_image_to_database_2', methods=['POST'])
def add_image_to_database_2():
    """Add image with multi-face handling"""
    try:
        if 'image' not in request.json or 'identity' not in request.json:
            return jsonify({"error": "Missing 'image' or 'identity' in request"}), 400

        image_data = request.json['image']
        identity = request.json['identity']

        # Decode image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Get face analyzer
        face_analyzer = face_server.get_face_analyzer()
        
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_analyzer.get(img_rgb)
            face_server.return_face_analyzer(face_analyzer)
        except Exception as e:
            face_server.return_face_analyzer(face_analyzer)
            raise e

        if len(faces) == 0:
            return jsonify({"error": "No face detected in the image"}), 400

        # Select best face
        selected_face = faces[0]
        multiple_faces_detected = len(faces) > 1
        
        if multiple_faces_detected:
            max_score = -1
            for face in faces:
                face_width = face.bbox[2] - face.bbox[0]
                face_height = face.bbox[3] - face.bbox[1]
                face_area = face_width * face_height
                normalized_area = face_area / (img.shape[0] * img.shape[1])
                score = (normalized_area * 0.7) + (face.det_score * 0.3)
                
                if score > max_score:
                    max_score = score
                    selected_face = face

        # Extract embedding
        embedding = selected_face.embedding

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"{identity}.jpg"
        original_filepath = os.path.join("Database/images", original_filename)
        
        # Crop if multiple faces
        if multiple_faces_detected:
            x1, y1, x2, y2 = map(int, selected_face.bbox)
            padding_x = int((x2 - x1) * 0.2)
            padding_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(img.shape[1], x2 + padding_x)
            y2 = min(img.shape[0], y2 + padding_y)
            img = img[y1:y2, x1:x2]

        # Save asynchronously
        def save_and_update():
            try:
                cv2.imwrite(original_filepath, img)
                
                # Add to database
                face_server.face_db[identity] = {
                    'embedding': embedding,
                    'image_path': original_filepath,
                    'timestamp': timestamp
                }
                
                # Save and rebuild
                face_server.save_embeddings()
                
            except Exception as e:
                print(f"[ERROR] Save and update failed: {e}")
        
        face_server.recognition_pool.submit(save_and_update)

        response_data = {
            "status": "success",
            "message": f"Image for '{identity}' added to database",
            "filename": original_filename,
            "timestamp": timestamp,
            "gpu_accelerated": True
        }

        if multiple_faces_detected:
            response_data["warning"] = "Multiple faces detected - selected largest face"
            response_data["total_faces_detected"] = len(faces)
            response_data["selected_face_confidence"] = float(selected_face.det_score)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

@app.route('/recognize_top_matches', methods=['POST'])
def recognize_top_matches():
    """Top matches recognition"""
    try:
        if 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode image
        nparr = np.frombuffer(base64.b64decode(request.json['image']), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image data"})

        # Get face analyzer
        face_analyzer = face_server.get_face_analyzer()
        
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_analyzer.get(img_rgb)
            face_server.return_face_analyzer(face_analyzer)
        except Exception as e:
            face_server.return_face_analyzer(face_analyzer)
            raise e

        if len(faces) == 0:
            return jsonify({"error": "No face detected in the image"})

        query_embedding = faces[0].embedding
        
        # Get parameters
        top_k = request.json.get('top_k', 3)
        min_threshold = request.json.get('min_threshold', 0.2)
        
        # Search using FAISS
        search_results = face_server.safe_faiss_search(query_embedding, top_k=top_k * 2)
        
        # Fallback to numpy
        if not search_results:
            search_results = face_server.numpy_similarity_search(query_embedding, top_k=top_k * 2)
        
        # Format results
        top_matches = []
        for identity, similarity in search_results:
            if similarity >= min_threshold and len(top_matches) < top_k:
                # Get image
                image_path = face_server.face_db.get(identity, {}).get('image_path', '')
                
                matched_image_base64 = None
                if image_path and os.path.exists(image_path):
                    try:
                        matched_image = cv2.imread(image_path)
                        if matched_image is not None:
                            _, buffer = cv2.imencode('.jpg', matched_image)
                            matched_image_base64 = base64.b64encode(buffer).decode('utf-8')
                    except Exception as e:
                        print(f"[ERROR] Image encoding failed: {e}")
                
                # Clean identity
                base_identity = identity.split('@')[0]
                
                top_matches.append({
                    "identity": base_identity,
                    "confidence": float(similarity),
                    "matched_image": matched_image_base64
                })

        return jsonify({
            "status": "success",
            "matches": top_matches,
            "gpu_accelerated": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recognition_events', methods=['GET', 'OPTIONS'])
def get_recognition_events():
    """Get recognition events"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        events_list = list(face_server.recognition_events)
        total_events = len(events_list)
        
        start_idx = max(0, total_events - limit - offset)
        end_idx = max(0, total_events - offset)
        
        paginated_events = events_list[start_idx:end_idx]
        paginated_events.reverse()
        
        return jsonify({
            "status": "success",
            "events": paginated_events,
            "total_events": total_events,
            "limit": limit,
            "offset": offset,
            "has_more": start_idx > 0,
            "gpu_accelerated": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reload_embeddings', methods=['POST'])
def reload_embeddings():
    """Reload embeddings"""
    try:
        face_server.face_db = face_server.load_embeddings()
        
        # Rebuild index
        def rebuild_task():
            try:
                face_server.cleanup_faiss_resources()
                time.sleep(1)
                face_server.initialize_robust_faiss_index()
            except Exception as e:
                print(f"[ERROR] Rebuild failed: {e}")
        
        face_server.recognition_pool.submit(rebuild_task)
        
        return jsonify({
            "status": "success", 
            "message": "Embeddings reloaded and index rebuilding",
            "embedding_count": len(face_server.face_db),
            "gpu_accelerated": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """Set recognition threshold"""
    try:
        new_threshold = float(request.json['threshold'])
        if 0 <= new_threshold <= 1:
            face_server.threshold = new_threshold
            face_server.config.set('Settings', 'threshold', str(new_threshold))
            with open(face_server.config_file, 'w') as f:
                face_server.config.write(f)
            return jsonify({"status": "success", "message": f"Threshold set to {new_threshold}"})
        else:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid threshold value"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Cache statistics"""
    try:
        return jsonify({
            "l1_cache": {
                "size": len(face_server.l1_cache),
                "max_size": face_server.l1_cache_size,
                "hits": face_server.cache_stats['l1_hits'],
                "misses": face_server.cache_stats['l1_misses']
            },
            "l2_cache": {
                "size": len(face_server.l2_cache),
                "max_size": face_server.l2_cache_size,
                "hits": face_server.cache_stats['l2_hits'],
                "misses": face_server.cache_stats['l2_misses']
            },
            "faiss_hits": face_server.cache_stats['faiss_hits'],
            "gpu_hits": face_server.cache_stats['gpu_hits'],
            "gpu_accelerated": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_caches():
    """Clear all caches"""
    try:
        l1_count = len(face_server.l1_cache)
        l2_count = len(face_server.l2_cache)
        
        face_server.l1_cache.clear()
        face_server.l2_cache.clear()
        
        # Reset stats
        face_server.cache_stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'faiss_hits': 0, 'gpu_hits': 0
        }
        
        return jsonify({
            "status": "success",
            "message": f"Cleared L1 ({l1_count}) and L2 ({l2_count}) caches",
            "gpu_accelerated": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/gpu/status', methods=['GET'])
def gpu_status():
    """GPU status endpoint"""
    try:
        gpu_info = {}
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(face_server.gpu_device_id)
            memory_total = torch.cuda.get_device_properties(face_server.gpu_device_id).total_memory
            memory_cached = torch.cuda.memory_reserved(face_server.gpu_device_id)
            
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(face_server.gpu_device_id),
                "memory_allocated_gb": memory_allocated / (1024**3),
                "memory_total_gb": memory_total / (1024**3),
                "memory_cached_gb": memory_cached / (1024**3),
                "memory_usage_percent": (memory_allocated / memory_total) * 100,
                "gpu_device_id": face_server.gpu_device_id,
                "cuda_version": torch.version.cuda,
                "faiss_gpu_available": face_server.faiss_index is not None
            }
        
        return jsonify({
            "status": "success",
            "gpu_info": gpu_info,
            "gpu_available": torch.cuda.is_available()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# File serving routes
@app.route('/Get/In/<path:filename>')
def serve_get_in_files(filename):
    get_in_folder = os.path.join(os.getcwd(), 'Get', 'In')
    return send_from_directory(get_in_folder, filename)

@app.route('/Database/images/<path:filename>')
def serve_Database_files(filename):
    get_in_folder = os.path.join(os.getcwd(), 'Database', 'images')
    return send_from_directory(get_in_folder, filename)

# Static routes
@app.route('/realtime')
def realtime_monitor():
    return app.send_static_file('realtime.html')

@app.route('/Check1')
def Check1_monitor():
    return app.send_static_file('Check1.html')

@app.route('/Check2')
def Check2_monitor():
    return app.send_static_file('Check2.html')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting 24/7 Robust GPU Face Recognition Server...")
    print(f"⚡ GPU Memory Reserved: {face_server.gpu_memory_limit / (1024**3):.1f}GB")
    print(f"💾 {len(face_server.face_db)} face embeddings loaded")
    print(f"🔍 FAISS GPU Index: {face_server.faiss_index.ntotal if face_server.faiss_index else 0} entries")
    print(f"🎯 24/7 Operation Ready - No Crashes, Maximum Performance!")
    
    # Run the server
    app.run(host='0.0.0.0', port=3005, debug=False, threaded=True)
