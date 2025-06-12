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
import shutil
import uuid
import queue
import threading
import time
import requests
from collections import deque, OrderedDict
import gc
import copy
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import multiprocessing as mp
import psutil
import torch
from functools import lru_cache
import weakref
import warnings

# Suppress the FutureWarning from insightface
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

# Optional imports with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] FAISS not available - using numpy similarity search")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("[WARNING] CuPy not available - using CPU arrays")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[WARNING] Redis not available - using local cache only")

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[WARNING] Numba not available - using standard numpy")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

class UltraFastFaceRecognitionServer:
    def __init__(self):
        self.config_file = 'config.ini'
        self.config = self.load_config()
        self.threshold = float(self.config.get('Settings', 'threshold', fallback=0.5))
        
        # ULTRA-HIGH PERFORMANCE CONFIGURATION
        self.num_cpu_cores = os.cpu_count()
        self.num_worker_threads = min(32, self.num_cpu_cores * 2)  # Reduced for stability
        self.num_http_workers = 8  # Reduced for stability
        self.num_gpu_streams = 4  # Reduced to prevent memory conflicts
        
        # BATCH PROCESSING (conservative sizes for stability)
        self.gpu_batch_size = 64  # Reduced from 256
        self.cpu_batch_size = 32  # Reduced from 128
        self.mega_batch_size = 256  # Reduced from 1024
        
        # Initialize Redis with proper error handling
        self.redis_client = None
        self.redis_enabled = False
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    db=0, 
                    decode_responses=False,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                    retry_on_timeout=False
                )
                # Test connection
                self.redis_client.ping()
                self.redis_enabled = True
                print("[SUCCESS] Redis connection established")
            except Exception as e:
                self.redis_enabled = False
                self.redis_client = None
                print(f"[INFO] Redis not available: {e}")
        
        # FACE ANALYZER POOL with thread safety
        self.face_analyzer_pool = []
        self.face_analyzer_lock = threading.Lock()
        self.initialize_face_analyzer_pool()
        
        # EMBEDDING DATABASE
        self.embeddings_file = 'Database/face_embeddings.pkl'
        self.face_db = self.load_embeddings()
        
        # FAISS INDEX with proper error handling
        self.faiss_index = None
        self.embedding_id_mapping = {}
        self.faiss_available = FAISS_AVAILABLE
        if self.faiss_available:
            self.initialize_faiss_index()
        
        # GPU-CACHED EMBEDDINGS with fallback
        self.gpu_embeddings = None
        self.gpu_available = CUPY_AVAILABLE
        if self.gpu_available:
            self.preload_embeddings_to_gpu()
        
        # MULTI-LEVEL CACHE SYSTEM
        self.l1_cache = {}  # Hash-based fast cache
        self.l2_cache = OrderedDict()  # Recent recognition cache
        self.l3_cache = {}  # Distributed cache placeholder
        
        # Cache configuration
        self.l1_cache_size = 5000  # Reduced for memory efficiency
        self.l2_cache_size = 20000  # Reduced for memory efficiency
        self.l2_cache_duration = 1800  # 30 minutes
        self.cache_similarity_threshold = 0.65
        
        # PERFORMANCE TRACKING
        self.cache_stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0,
            'gpu_hits': 0, 'faiss_hits': 0
        }
        
        # DIRECTORIES
        os.makedirs("Get/In", exist_ok=True)
        os.makedirs("Get/Out", exist_ok=True)
        os.makedirs("static/images", exist_ok=True)
        os.makedirs("static/varient", exist_ok=True)
        os.makedirs("Logs", exist_ok=True)
        
        # API ENDPOINTS
        self.log_api_endpoint = "http://192.168.14.102:7578/api/FaceRecognition/Recognize-Logs"
        self.log_api_endpoint_2 = "http://192.168.15.129:5002/add_log"
        self.http_broadcast_endpoints = [
            "http://192.168.14.102:7578/api/FaceRecognition/Recognition-Events",
            "http://192.168.15.129:5002/recognition_event"
        ]
        
        # QUEUES with reasonable sizes
        self.ultra_priority_queue = queue.PriorityQueue(maxsize=500)
        self.high_priority_queue = queue.PriorityQueue(maxsize=1000)
        self.normal_priority_queue = queue.PriorityQueue(maxsize=2000)
        self.batch_request_queue = queue.PriorityQueue(maxsize=200)
        self.mega_batch_queue = queue.PriorityQueue(maxsize=50)
        
        # RESPONSE CACHES
        self.response_cache = {}
        self.batch_response_cache = {}
        self.recognition_events = deque(maxlen=5000)
        self.http_broadcast_queue = queue.Queue(maxsize=2000)
        
        # PERFORMANCE METRICS
        self.processed_count = 0
        self.start_time = time.time()
        self.request_times = deque(maxlen=5000)
        
        # THREAD POOLS with conservative sizes
        self.recognition_pool = ThreadPoolExecutor(max_workers=self.num_worker_threads)
        self.gpu_pool = ThreadPoolExecutor(max_workers=self.num_gpu_streams)
        self.batch_processing_pool = ThreadPoolExecutor(max_workers=4)  # Reduced
        self.http_pool = ThreadPoolExecutor(max_workers=self.num_http_workers)
        self.mega_batch_pool = ThreadPoolExecutor(max_workers=2)  # Reduced
        
        # START WORKER SYSTEMS
        self.start_all_workers()
        
        print(f"[ULTRA-FAST] Server initialized with:")
        print(f"  - {self.num_worker_threads} worker threads")
        print(f"  - {self.num_gpu_streams} GPU streams")
        print(f"  - GPU batch size: {self.gpu_batch_size}")
        print(f"  - FAISS available: {self.faiss_available}")
        print(f"  - FAISS index: {self.faiss_index.ntotal if self.faiss_index else 0} embeddings")
        print(f"  - GPU available: {self.gpu_available}")
        print(f"  - Redis: {'Enabled' if self.redis_enabled else 'Disabled'}")

    def initialize_face_analyzer_pool(self):
        """Initialize pool of face analyzers with error handling"""
        print("[INIT] Creating face analyzer pool...")
        
        pool_size = min(8, self.num_worker_threads // 4)  # Reduced pool size
        
        for i in range(pool_size):
            try:
                analyzer = FaceAnalysis(
                    name='buffalo_l',
                    root='models',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                # Use different GPU contexts to avoid conflicts
                ctx_id = i % 2 if torch.cuda.device_count() > 1 else 0
                analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self.face_analyzer_pool.append(analyzer)
                print(f"[INIT] Created analyzer {i+1}/{pool_size} on GPU context {ctx_id}")
            except Exception as e:
                print(f"[WARNING] Failed to create analyzer {i}: {e}")
                # Try CPU fallback
                try:
                    analyzer = FaceAnalysis(
                        name='buffalo_l',
                        root='models',
                        providers=['CPUExecutionProvider']
                    )
                    analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                    self.face_analyzer_pool.append(analyzer)
                    print(f"[INIT] Created CPU analyzer {i+1}/{pool_size}")
                except Exception as e2:
                    print(f"[ERROR] Failed to create CPU analyzer {i}: {e2}")
                    
        print(f"[INIT] Created {len(self.face_analyzer_pool)} face analyzers")

    def get_face_analyzer(self):
        """Get face analyzer from pool with timeout"""
        timeout = 5.0  # 5 second timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.face_analyzer_lock:
                if self.face_analyzer_pool:
                    return self.face_analyzer_pool.pop()
            time.sleep(0.01)
        
        # Fallback: create temporary analyzer
        print("[WARNING] Creating temporary face analyzer - pool exhausted")
        try:
            analyzer = FaceAnalysis(
                name='buffalo_l',
                root='models',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            analyzer.prepare(ctx_id=0, det_size=(640, 640))
            return analyzer
        except Exception as e:
            print(f"[ERROR] Failed to create temporary analyzer: {e}")
            # Try CPU fallback
            analyzer = FaceAnalysis(
                name='buffalo_l',
                root='models',
                providers=['CPUExecutionProvider']
            )
            analyzer.prepare(ctx_id=-1, det_size=(640, 640))
            return analyzer

    def return_face_analyzer(self, analyzer):
        """Return face analyzer to pool"""
        with self.face_analyzer_lock:
            if len(self.face_analyzer_pool) < 8:  # Limit pool size
                self.face_analyzer_pool.append(analyzer)
            # If pool is full, let the analyzer be garbage collected

    def initialize_faiss_index(self):
        """Initialize FAISS index with proper GPU memory management"""
        if not self.faiss_available:
            print("[INFO] FAISS not available, using numpy similarity search")
            return
            
        print("[INIT] Building FAISS GPU index...")
        
        if not self.face_db:
            print("[WARNING] No embeddings to index")
            return
        
        try:
            # Extract all embeddings
            embeddings = []
            identities = []
            
            for identity, data in self.face_db.items():
                if 'embedding' in data and data['embedding'] is not None:
                    embeddings.append(data['embedding'])
                    identities.append(identity)
            
            if not embeddings:
                print("[WARNING] No valid embeddings found")
                return
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_array = embeddings_array / norms
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            
            # CRITICAL FIX: Use proper GPU resource management to avoid memory corruption
            if faiss.get_num_gpus() > 0:
                try:
                    # Method 1: Use multiple GPU resources to avoid memory conflicts
                    self.gpu_resources = []
                    self.faiss_indices = []
                    
                    # Create separate GPU resources for different operations
                    for i in range(min(2, faiss.get_num_gpus())):  # Use max 2 GPUs
                        gpu_res = faiss.StandardGpuResources()
                        
                        # CRITICAL: Set proper memory limits to prevent corruption
                        # Reserve more memory but manage it properly
                        gpu_res.setTempMemory(512 * 1024 * 1024)  # 512MB per GPU resource
                        gpu_res.setDefaultNullStreamAllDevices()
                        
                        self.gpu_resources.append(gpu_res)
                    
                    # Create GPU index with the first GPU resource
                    cpu_index = faiss.IndexFlatIP(dimension)
                    
                    # CRITICAL: Use GpuIndexFlatConfig to control memory allocation
                    config = faiss.GpuIndexFlatConfig()
                    config.useFloat16 = False  # Use full precision to avoid issues
                    config.device = 0  # Use GPU 0
                    
                    # Create GPU index directly instead of transferring
                    self.faiss_index = faiss.GpuIndexFlatIP(
                        self.gpu_resources[0], 
                        dimension, 
                        config
                    )
                    
                    print("[INIT] FAISS GPU index created with proper memory management")
                    
                except Exception as e:
                    print(f"[ERROR] GPU FAISS creation failed: {e}")
                    # Fallback to a more conservative approach
                    try:
                        # Method 2: Use single GPU resource with minimal memory
                        self.gpu_resources = [faiss.StandardGpuResources()]
                        self.gpu_resources[0].setTempMemory(128 * 1024 * 1024)  # 128MB only
                        
                        cpu_index = faiss.IndexFlatIP(dimension)
                        self.faiss_index = faiss.index_cpu_to_gpu(
                            self.gpu_resources[0], 0, cpu_index
                        )
                        print("[INIT] FAISS GPU index created with minimal memory")
                        
                    except Exception as e2:
                        print(f"[ERROR] All GPU FAISS methods failed: {e2}")
                        # Last resort: Force CPU usage
                        self.faiss_index = faiss.IndexFlatIP(dimension)
                        print("[INIT] Using CPU FAISS index as fallback")
            else:
                self.faiss_index = faiss.IndexFlatIP(dimension)
                print("[INIT] No GPU available, using CPU FAISS index")
            
            # Add embeddings to index in batches to prevent memory issues
            batch_size = 1000  # Process 1000 embeddings at a time
            total_added = 0
            
            for i in range(0, len(embeddings_array), batch_size):
                batch_embeddings = embeddings_array[i:i + batch_size]
                self.faiss_index.add(batch_embeddings)
                total_added += len(batch_embeddings)
                
                # Force synchronization to prevent memory corruption
                if hasattr(self.faiss_index, 'sync'):
                    self.faiss_index.sync()
                
                print(f"[INIT] Added {total_added}/{len(embeddings_array)} embeddings to FAISS index")
            
            # Create mapping from FAISS index to identity
            self.embedding_id_mapping = {i: identities[i] for i in range(len(identities))}
            
            print(f"[INIT] FAISS GPU index built successfully with {self.faiss_index.ntotal} embeddings")
            
        except Exception as e:
            print(f"[ERROR] Failed to build FAISS index: {e}")
            self.faiss_index = None
            self.faiss_available = False
            # Clean up GPU resources if they were created
            if hasattr(self, 'gpu_resources'):
                del self.gpu_resources

    def preload_embeddings_to_gpu(self):
        """Preload embeddings to GPU memory with error handling"""
        if not self.gpu_available or not self.face_db:
            return
        
        try:
            embeddings = []
            for identity, data in self.face_db.items():
                if 'embedding' in data and data['embedding'] is not None:
                    embeddings.append(data['embedding'])
            
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                # Check available GPU memory before allocation
                mempool = cp.get_default_memory_pool()
                available_memory = mempool.free_bytes()
                required_memory = embeddings_array.nbytes
                
                if available_memory > required_memory * 2:  # Keep 50% free memory
                    self.gpu_embeddings = cp.asarray(embeddings_array)
                    print(f"[INIT] Preloaded {len(embeddings)} embeddings to GPU memory")
                else:
                    print(f"[WARNING] Insufficient GPU memory for embedding preload")
                    
        except Exception as e:
            print(f"[WARNING] Failed to preload embeddings to GPU: {e}")
            self.gpu_embeddings = None

    def ultra_fast_similarity_search(self, query_embedding, top_k=1):
        """Ultra-fast similarity search with proper GPU memory management"""
        try:
            if self.faiss_available and self.faiss_index is not None:
                # Use FAISS for ultra-fast search with proper synchronization
                query_embedding = query_embedding.astype(np.float32)
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
                query_embedding = query_embedding.reshape(1, -1)
                
                # CRITICAL: Use proper synchronization to prevent memory corruption
                try:
                    # Create a copy of the query to avoid memory conflicts
                    query_copy = np.copy(query_embedding)
                    
                    # Use synchronous search to prevent memory corruption
                    with threading.Lock():  # Ensure thread safety for GPU operations
                        similarities, indices = self.faiss_index.search(query_copy, top_k)
                        
                        # Force synchronization if available
                        if hasattr(self.faiss_index, 'sync'):
                            self.faiss_index.sync()
                    
                    results = []
                    for i, (similarity, index) in enumerate(zip(similarities[0], indices[0])):
                        if index != -1 and index < len(self.embedding_id_mapping):
                            identity = self.embedding_id_mapping[index]
                            results.append((identity, float(similarity)))
                    
                    self.cache_stats['faiss_hits'] += 1
                    return results
                    
                except Exception as e:
                    print(f"[ERROR] FAISS search failed, using numpy fallback: {e}")
                    # If FAISS search fails, recreate the index
                    self.faiss_recovery_count = getattr(self, 'faiss_recovery_count', 0) + 1
                    if self.faiss_recovery_count < 3:  # Try to recover up to 3 times
                        print(f"[RECOVERY] Attempting FAISS recovery #{self.faiss_recovery_count}")
                        threading.Thread(target=self.recover_faiss_index, daemon=True).start()
                    # Fall back to numpy search
                    pass
            
            # Fallback to numpy-based similarity search
            return self.numpy_similarity_search(query_embedding, top_k)
            
        except Exception as e:
            print(f"[ERROR] Similarity search failed: {e}")
            return []

    def recover_faiss_index(self):
        """Recover FAISS index after corruption"""
        try:
            print("[RECOVERY] Attempting to recover FAISS index...")
            
            # Clean up existing resources
            if hasattr(self, 'gpu_resources'):
                try:
                    del self.gpu_resources
                except:
                    pass
            
            if hasattr(self, 'faiss_index'):
                try:
                    del self.faiss_index
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
            
            # Wait for GPU memory to be released
            import time
            time.sleep(2)
            
            # Rebuild the index
            self.initialize_faiss_index()
            
            if self.faiss_index is not None:
                print("[RECOVERY] FAISS index recovered successfully")
            else:
                print("[RECOVERY] FAISS recovery failed, will use numpy fallback")
                
        except Exception as e:
            print(f"[ERROR] FAISS recovery failed: {e}")

    def safe_faiss_search(self, query_embedding, top_k=1):
        """Thread-safe FAISS search with memory protection"""
        try:
            # Use a global lock for all FAISS operations to prevent memory corruption
            if not hasattr(UltraFastFaceRecognitionServer, '_faiss_lock'):
                UltraFastFaceRecognitionServer._faiss_lock = threading.RLock()
            
            with UltraFastFaceRecognitionServer._faiss_lock:
                if self.faiss_index is None:
                    return []
                
                # Prepare query
                query_embedding = query_embedding.astype(np.float32)
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
                query_embedding = query_embedding.reshape(1, -1)
                
                # Create isolated copy to prevent memory conflicts
                query_isolated = np.copy(query_embedding)
                
                # Perform search
                similarities, indices = self.faiss_index.search(query_isolated, top_k)
                
                # Process results immediately
                results = []
                for i, (similarity, index) in enumerate(zip(similarities[0], indices[0])):
                    if index != -1 and index < len(self.embedding_id_mapping):
                        identity = self.embedding_id_mapping[index]
                        results.append((identity, float(similarity)))
                
                return results
                
        except Exception as e:
            print(f"[ERROR] Safe FAISS search failed: {e}")
            return []

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
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"[ERROR] Numpy similarity search failed: {e}")
            return []

    def check_l1_cache(self, query_embedding):
        """Check L1 cache with error handling"""
        try:
            # Create a simple hash of the embedding
            embedding_bytes = query_embedding.astype(np.float32).tobytes()
            embedding_hash = hash(embedding_bytes[:64])  # Use first 64 bytes for hash
            
            if embedding_hash in self.l1_cache:
                result, timestamp = self.l1_cache[embedding_hash]
                
                # Check if cache entry is still valid (5 minutes)
                if time.time() - timestamp < 300:
                    self.cache_stats['l1_hits'] += 1
                    result = result.copy()
                    result['l1_cache_hit'] = True
                    return result
                else:
                    # Remove expired entry
                    del self.l1_cache[embedding_hash]
            
            self.cache_stats['l1_misses'] += 1
            return None
            
        except Exception as e:
            print(f"[ERROR] L1 cache check failed: {e}")
            self.cache_stats['l1_misses'] += 1
            return None

    def add_to_l1_cache(self, query_embedding, result):
        """Add result to L1 cache with error handling"""
        try:
            if result.get('status') == 'matched' and result.get('confidence', 0) > 0.6:
                embedding_bytes = query_embedding.astype(np.float32).tobytes()
                embedding_hash = hash(embedding_bytes[:64])
                
                # Limit cache size
                if len(self.l1_cache) >= self.l1_cache_size:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(self.l1_cache.keys())[:100]
                    for key in oldest_keys:
                        if key in self.l1_cache:
                            del self.l1_cache[key]
                
                self.l1_cache[embedding_hash] = (result.copy(), time.time())
                
        except Exception as e:
            print(f"[ERROR] L1 cache add failed: {e}")

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
            
            # Check against recent embeddings
            expired_keys = []
            for cache_key, (cached_embedding, cached_result, timestamp) in list(self.l2_cache.items()):
                # Remove expired entries
                if current_time - timestamp > self.l2_cache_duration:
                    expired_keys.append(cache_key)
                    continue
                
                # Calculate similarity
                try:
                    similarity = float(np.dot(normalized_query, cached_embedding))
                    
                    if similarity > self.cache_similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = cached_result.copy()
                        best_match['l2_cache_hit'] = True
                        best_match['l2_cache_similarity'] = similarity
                        best_match['cache_age_seconds'] = current_time - timestamp
                except Exception as e:
                    print(f"[ERROR] L2 similarity calculation failed: {e}")
                    continue
            
            # Clean up expired entries
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
            print(f"[ERROR] L2 cache check failed: {e}")
            self.cache_stats['l2_misses'] += 1
            return None

    def add_to_l2_cache(self, query_embedding, result):
        """Add result to L2 cache"""
        try:
            if result.get('status') == 'matched' and result.get('confidence', 0) > 0.5:
                # Limit cache size
                if len(self.l2_cache) >= self.l2_cache_size:
                    # Remove oldest entries
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
            print(f"[ERROR] L2 cache add failed: {e}")

    def check_l3_cache(self, query_embedding):
        """Check L3 cache (Redis) with proper error handling"""
        if not self.redis_enabled or not self.redis_client:
            self.cache_stats['l3_misses'] += 1
            return None
        
        try:
            # Create cache key from embedding
            embedding_bytes = query_embedding.astype(np.float32).tobytes()
            embedding_hash = hash(embedding_bytes[:64])
            cache_key = f"face_cache:{embedding_hash}"
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                result = pickle.loads(cached_data)
                self.cache_stats['l3_hits'] += 1
                result['l3_cache_hit'] = True
                return result
            
            self.cache_stats['l3_misses'] += 1
            return None
            
        except Exception as e:
            # Don't log Redis connection errors every time
            if "Connection refused" not in str(e):
                print(f"[ERROR] L3 cache check failed: {e}")
            self.cache_stats['l3_misses'] += 1
            return None

    def add_to_l3_cache(self, query_embedding, result):
        """Add result to L3 cache (Redis) with error handling"""
        if not self.redis_enabled or not self.redis_client:
            return
        
        try:
            if result.get('status') == 'matched' and result.get('confidence', 0) > 0.6:
                embedding_bytes = query_embedding.astype(np.float32).tobytes()
                embedding_hash = hash(embedding_bytes[:64])
                cache_key = f"face_cache:{embedding_hash}"
                
                # Cache for 30 minutes
                self.redis_client.setex(
                    cache_key, 
                    1800, 
                    pickle.dumps(result.copy())
                )
                
        except Exception as e:
            # Don't log Redis connection errors every time
            if "Connection refused" not in str(e):
                print(f"[ERROR] L3 cache add failed: {e}")

    def ultra_fast_recognize_face(self, image_data, camera_id=None):
        """Ultra-fast face recognition with robust FAISS GPU handling"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_id = str(uuid.uuid4())[:8]
        
        try:
            # Convert base64 to image
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"status": "error", "message": "Invalid image data"}

            # Prepare camera ID prefix
            prefix = f"cam{camera_id}_{request_id}_" if camera_id else f"{request_id}_"
            
            # Save input image asynchronously
            input_filename = self.save_image_async(img, "Get/In", f"{prefix}input")

            # Get face analyzer from pool
            face_analyzer = self.get_face_analyzer()
            
            try:
                # Convert to RGB for face analysis
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_analyzer.get(img_rgb)
                
                # Return analyzer to pool immediately
                self.return_face_analyzer(face_analyzer)
                
            except Exception as e:
                self.return_face_analyzer(face_analyzer)
                raise e

            if len(faces) == 0:
                # No face detected
                return self.handle_no_face_detected(input_filename, prefix, camera_id, request_id)

            # Extract query embedding
            query_embedding = faces[0].embedding

            # MULTI-LEVEL CACHE CHECK (L1 -> L2 -> L3)
            
            # L1 Cache Check (fastest)
            cached_result = self.check_l1_cache(query_embedding)
            if cached_result:
                cached_result.update({
                    "input_filename": input_filename,
                    "request_id": request_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp
                })
                return cached_result

            # L2 Cache Check (fast)
            cached_result = self.check_l2_cache(query_embedding)
            if cached_result:
                cached_result.update({
                    "input_filename": input_filename,
                    "request_id": request_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp
                })
                
                # Promote to L1 cache
                self.add_to_l1_cache(query_embedding, cached_result)
                return cached_result

            # L3 Cache Check (distributed)
            cached_result = self.check_l3_cache(query_embedding)
            if cached_result:
                cached_result.update({
                    "input_filename": input_filename,
                    "request_id": request_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp
                })
                
                # Promote to L2 and L1 caches
                self.add_to_l2_cache(query_embedding, cached_result)
                self.add_to_l1_cache(query_embedding, cached_result)
                return cached_result

            # DATABASE SEARCH using robust FAISS or numpy fallback
            search_results = self.safe_faiss_search(query_embedding, top_k=1)
            
            # If FAISS failed, use the original method as backup
            if not search_results:
                search_results = self.ultra_fast_similarity_search(query_embedding, top_k=1)
            
            if search_results:
                best_identity, best_similarity = search_results[0]
                
                if best_similarity > self.threshold:
                    # Found match
                    result = self.create_match_result(
                        best_identity, best_similarity, input_filename, 
                        prefix, camera_id, request_id, query_embedding
                    )
                    
                    # Add to all cache levels
                    self.add_to_l1_cache(query_embedding, result)
                    self.add_to_l2_cache(query_embedding, result)
                    self.add_to_l3_cache(query_embedding, result)
                    
                    return result
            
            # No match found
            return self.handle_unrecognized_face(
                img, faces[0], input_filename, prefix, camera_id, request_id
            )
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during face recognition: {str(e)}",
                "camera_id": camera_id,
                "request_id": request_id
            }

    def save_image_async(self, img, folder, prefix):
        """Save image asynchronously with error handling"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        
        def save_task():
            try:
                cv2.imwrite(filepath, img)
            except Exception as e:
                print(f"[ERROR] Failed to save image {filepath}: {e}")
        
        # Submit to thread pool but don't wait
        self.recognition_pool.submit(save_task)
        return filename

    def handle_no_face_detected(self, input_filename, prefix, camera_id, request_id):
        """Handle case where no face is detected"""
        try:
            unrecognized_img = cv2.imread("Database/UNRECOGNIZED.png")
            if unrecognized_img is None:
                # Create a simple placeholder image if UNRECOGNIZED.png doesn't exist
                unrecognized_img = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.putText(unrecognized_img, "NO FACE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            output_filename = self.save_image_async(unrecognized_img, "Get/Out", f"{prefix}unrecognized")
            _, buffer = cv2.imencode('.jpg', unrecognized_img)
            unrecognized_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "status": "unrecognized",
                "message": "No face detected",
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_image": unrecognized_image_base64,
                "camera_id": camera_id,
                "request_id": request_id
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error handling no face detected: {str(e)}",
                "input_filename": input_filename,
                "camera_id": camera_id,
                "request_id": request_id
            }

    def create_match_result(self, identity, similarity, input_filename, prefix, camera_id, request_id, query_embedding):
        """Create result for successful match with error handling"""
        try:
            # Find image path for identity
            image_path = None
            for face_identity, data in self.face_db.items():
                if face_identity.split('@')[0] == identity.split('@')[0]:
                    image_path = data.get('image_path')
                    break
            
            if not image_path or not os.path.exists(image_path):
                # Use a default image or create placeholder
                image_path = "Database/UNRECOGNIZED.png"
                if not os.path.exists(image_path):
                    # Create placeholder
                    placeholder_img = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.putText(placeholder_img, "MATCHED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    matched_image = placeholder_img
                else:
                    matched_image = cv2.imread(image_path)
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
                "matched_filename": os.path.basename(image_path),
                "camera_id": camera_id,
                "request_id": request_id,
                "ultra_fast": True
            }
            
            # Log recognition details asynchronously
            self.log_recognition_details_async(result, camera_id)
            
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
        """Handle unrecognized face with error handling"""
        try:
            saved_path, face_id = self.save_unrecognized_face_async(img, face, input_filename, camera_id)
            
            # Create a copy of the input image for output
            output_filename = self.save_image_async(img, "Get/Out", f"{prefix}unrecognized")
            
            _, buffer = cv2.imencode('.jpg', img)
            unrecognized_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "status": "unrecognized_saved",
                "message": "Face not recognized - saved to database",
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_image": unrecognized_image_base64,
                "face_id": face_id,
                "saved_path": saved_path,
                "camera_id": camera_id,
                "request_id": request_id
            }
            
            # Log recognition details asynchronously
            self.log_recognition_details_async(result, camera_id)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error handling unrecognized face: {str(e)}",
                "input_filename": input_filename,
                "camera_id": camera_id,
                "request_id": request_id
            }

    def save_unrecognized_face_async(self, img, face, input_filename, camera_id=None):
        """Save unrecognized face asynchronously"""
        face_id = os.path.splitext(input_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"unrecognized_{os.path.splitext(input_filename)[0]}_{timestamp}.jpg"
        filepath = os.path.join("static/images", filename)
        
        def save_task():
            try:
                cv2.imwrite(filepath, img)
            except Exception as e:
                print(f"[ERROR] Failed to save unrecognized face {filepath}: {e}")
        
        self.recognition_pool.submit(save_task)
        return filepath, face_id

    def log_recognition_details_async(self, result, camera_id=None):
        """Log recognition details asynchronously"""
        def log_task():
            try:
                # Prepare HTTP broadcast event data
                http_event_data = {
                    "timestamp": datetime.now().isoformat(),
                    "status": result.get('status', 'Unknown'),
                    "camera_id": camera_id,
                    "camera_name": self.get_camera_name(camera_id),
                    "confidence": result.get('confidence', 0.0),
                    "matched_filename": result.get('matched_filename', ''),
                    "identity": os.path.splitext(result.get('matched_filename', ''))[0] if result.get('matched_filename') else None,
                    "input_image": f"/Get/In/{result.get('input_filename', '')}" if result.get('input_filename') else None,
                    "matched_image": f"Database/images/{result.get('matched_filename', '')}" if result.get('matched_filename') else None,
                    "cache_hit": result.get('l1_cache_hit', False) or result.get('l2_cache_hit', False) or result.get('l3_cache_hit', False),
                    "request_id": result.get('request_id', ''),
                }

                # Add to events and broadcast
                self.recognition_events.append(http_event_data)
                
                # Add to broadcast queue (non-blocking)
                try:
                    self.http_broadcast_queue.put_nowait(http_event_data)
                except queue.Full:
                    # If queue is full, skip this broadcast
                    pass

                # Send to API endpoints
                self._send_log_to_apis_async(result, camera_id)
                    
            except Exception as e:
                print(f"[ERROR] Logging recognition details: {e}")
        
        self.recognition_pool.submit(log_task)

    def _send_log_to_apis_async(self, result, camera_id):
        """Send logs to API endpoints asynchronously"""
        def api_log_task():
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
                    "cache_hit": result.get('l1_cache_hit', False) or result.get('l2_cache_hit', False) or result.get('l3_cache_hit', False)
                }

                api_endpoints = [self.log_api_endpoint, self.log_api_endpoint_2]

                for endpoint in api_endpoints:
                    try:
                        if endpoint == self.log_api_endpoint_2:
                            log_entry["camera_id"] = camera_id

                        response = requests.post(
                            endpoint,
                            json=[log_entry],
                            headers={'Content-Type': 'application/json'},
                            timeout=1.0
                        )
                    except Exception as e:
                        # Don't log API failures to avoid spam
                        pass
            except Exception as e:
                print(f"[ERROR] API logging: {e}")
        
        self.recognition_pool.submit(api_log_task)

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

    def start_all_workers(self):
        """Start all worker threads and processes with error handling"""
        
        # Ultra-high priority queue workers
        for i in range(2):  # Reduced
            thread = threading.Thread(target=self.process_ultra_priority_queue, daemon=True)
            thread.start()
        
        # High priority queue workers
        for i in range(4):  # Reduced
            thread = threading.Thread(target=self.process_high_priority_queue, daemon=True)
            thread.start()
        
        # Normal priority queue workers
        for i in range(min(8, self.num_worker_threads)):  # Reduced
            thread = threading.Thread(target=self.process_normal_priority_queue, daemon=True)
            thread.start()
        
        # Batch processing workers
        for i in range(2):  # Reduced
            thread = threading.Thread(target=self.process_batch_queue_ultra, daemon=True)
            thread.start()
        
        # Mega batch processing workers
        for i in range(1):  # Reduced
            thread = threading.Thread(target=self.process_mega_batch_queue, daemon=True)
            thread.start()
        
        # HTTP broadcast workers
        for i in range(min(4, self.num_http_workers)):  # Reduced
            thread = threading.Thread(target=self.process_http_broadcasts_ultra, daemon=True)
            thread.start()
        
        # Cache maintenance worker
        thread = threading.Thread(target=self.cache_maintenance_worker, daemon=True)
        thread.start()
        
        # Performance monitoring worker
        thread = threading.Thread(target=self.performance_monitor_worker, daemon=True)
        thread.start()

    def process_ultra_priority_queue(self):
        """Process ultra-priority requests with error handling"""
        while True:
            try:
                _, (request_id, image_data, camera_id) = self.ultra_priority_queue.get(block=True, timeout=0.1)
                
                start_time = time.time()
                result = self.ultra_fast_recognize_face(image_data, camera_id)
                processing_time = time.time() - start_time
                
                # Store result
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
        """Process high-priority requests with error handling"""
        while True:
            try:
                _, (request_id, image_data, camera_id) = self.high_priority_queue.get(block=True, timeout=0.1)
                
                start_time = time.time()
                result = self.ultra_fast_recognize_face(image_data, camera_id)
                processing_time = time.time() - start_time
                
                # Store result
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
        """Process normal priority requests with error handling"""
        while True:
            try:
                _, (request_id, image_data, camera_id) = self.normal_priority_queue.get(block=True, timeout=0.1)
                
                start_time = time.time()
                result = self.ultra_fast_recognize_face(image_data, camera_id)
                processing_time = time.time() - start_time
                
                # Store result
                self.response_cache[request_id] = (result, time.time())
                self.processed_count += 1
                self.request_times.append(processing_time)
                
                self.normal_priority_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] Normal priority queue: {e}")
                time.sleep(0.1)

    def process_batch_queue_ultra(self):
        """Ultra-fast batch processing with error handling"""
        while True:
            try:
                _, (request_id, files_to_process) = self.batch_request_queue.get()
                
                print(f"[BATCH-ULTRA] Processing {len(files_to_process)} files")
                
                # Use batch processing
                results = self.batch_process_files_safe(files_to_process)
                
                # Store results
                self.batch_response_cache[request_id] = (results, time.time())
                self.batch_request_queue.task_done()
                
            except Exception as e:
                print(f"[ERROR] Batch queue ultra: {e}")
                time.sleep(0.1)

    def batch_process_files_safe(self, files_to_process):
        """Safe batch processing with comprehensive error handling"""
        results = []
        batch_size = min(32, len(files_to_process))  # Smaller batches for stability
        
        # Process in smaller batches
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]
            
            # Use ThreadPoolExecutor for CPU-bound batch processing
            with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
                futures = []
                for filename, image_id in batch:
                    future = executor.submit(self.process_batch_file_safe, filename, image_id)
                    futures.append(future)
                
                # Collect results with timeout
                for future in concurrent.futures.as_completed(futures, timeout=30):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"[BATCH] Error processing file: {e}")
                        results.append({
                            "status": "error",
                            "message": str(e)
                        })
            
            # Brief pause between batches
            time.sleep(0.01)
        
        return results

    def process_batch_file_safe(self, filename, image_id):
        """Process a single batch file with comprehensive error handling"""
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
            
            # Read and process image
            img = cv2.imread(image_path)
            if img is None:
                return {
                    "image_id": image_id,
                    "status": "error",
                    "message": "Invalid image file",
                    "request_id": request_id
                }
            
            # Convert to base64 for processing
            _, buffer = cv2.imencode('.jpg', img)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Use the main recognition function
            result = self.ultra_fast_recognize_face(image_data)
            
            # Format result for batch response
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
            elif result.get('status') in ['unrecognized', 'unrecognized_saved']:
                batch_result["message"] = result.get('message', 'No match found')
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

    def process_mega_batch_queue(self):
        """Process mega batch requests with error handling"""
        while True:
            try:
                _, (request_id, files_to_process) = self.mega_batch_queue.get()
                
                print(f"[MEGA-BATCH] Processing {len(files_to_process)} files")
                
                # Use mega batch processing
                results = self.mega_batch_process_safe(files_to_process)
                
                # Store results
                self.batch_response_cache[request_id] = (results, time.time())
                self.mega_batch_queue.task_done()
                
            except Exception as e:
                print(f"[ERROR] Mega batch queue: {e}")
                time.sleep(0.1)

    def mega_batch_process_safe(self, files_list, batch_size=None):
        """Safe mega batch processing"""
        if batch_size is None:
            batch_size = min(128, len(files_list))  # Smaller mega batches
        
        results = []
        total_files = len(files_list)
        
        print(f"[MEGA-BATCH] Processing {total_files} files in batches of {batch_size}")
        
        # Process in chunks
        for i in range(0, total_files, batch_size):
            batch = files_list[i:i + batch_size]
            
            # Use the regular batch processing for each chunk
            batch_results = self.batch_process_files_safe(batch)
            results.extend(batch_results)
            
            # Progress update
            progress = min(i + batch_size, total_files)
            print(f"[MEGA-BATCH] Progress: {progress}/{total_files} ({progress/total_files*100:.1f}%)")
            
            # Brief pause to prevent system overload
            time.sleep(0.1)
        
        return results

    def process_http_broadcasts_ultra(self):
        """Ultra-fast HTTP broadcast processing with error handling"""
        while True:
            try:
                event_data = self.http_broadcast_queue.get(timeout=1.0)
                
                # Use thread pool for non-blocking HTTP requests
                self.http_pool.submit(self._send_http_broadcast_ultra, event_data)
                
                self.http_broadcast_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] HTTP broadcast ultra: {e}")
                time.sleep(0.1)

    def _send_http_broadcast_ultra(self, event_data):
        """Send HTTP broadcast with ultra-short timeout and error handling"""
        for endpoint in self.http_broadcast_endpoints:
            try:
                response = requests.post(
                    endpoint,
                    json=event_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=0.5
                )
            except Exception:
                # Ignore all HTTP broadcast failures to maintain performance
                pass

    def cache_maintenance_worker(self):
        """Maintain all cache levels for optimal performance"""
        while True:
            try:
                time.sleep(30)  # Run every 30 seconds
                
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
                    if current_time - timestamp > 120  # 2 minutes
                ]
                for key in expired_response_keys:
                    if key in self.response_cache:
                        del self.response_cache[key]
                
                expired_batch_keys = [
                    key for key, (_, timestamp) in list(self.batch_response_cache.items())
                    if current_time - timestamp > 600  # 10 minutes
                ]
                for key in expired_batch_keys:
                    if key in self.batch_response_cache:
                        del self.batch_response_cache[key]
                
                # Force garbage collection periodically
                if self.processed_count % 1000 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"[ERROR] Cache maintenance: {e}")

    def performance_monitor_worker(self):
        """Monitor and log performance metrics"""
        while True:
            try:
                time.sleep(60)  # Report every 60 seconds
                
                uptime = time.time() - self.start_time
                rps = self.processed_count / uptime if uptime > 0 else 0
                
                avg_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
                
                # Calculate cache hit rates
                total_l1 = self.cache_stats['l1_hits'] + self.cache_stats['l1_misses']
                total_l2 = self.cache_stats['l2_hits'] + self.cache_stats['l2_misses']
                total_l3 = self.cache_stats['l3_hits'] + self.cache_stats['l3_misses']
                
                l1_rate = (self.cache_stats['l1_hits'] / total_l1 * 100) if total_l1 > 0 else 0
                l2_rate = (self.cache_stats['l2_hits'] / total_l2 * 100) if total_l2 > 0 else 0
                l3_rate = (self.cache_stats['l3_hits'] / total_l3 * 100) if total_l3 > 0 else 0
                
                print(f"[PERF] RPS: {rps:.1f} | Avg: {avg_time*1000:.1f}ms | "
                      f"L1: {l1_rate:.1f}% | L2: {l2_rate:.1f}% | L3: {l3_rate:.1f}%")
                
            except Exception as e:
                print(f"[ERROR] Performance monitor: {e}")

    # COMPATIBILITY METHODS
    
    def load_config(self):
        config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            config['Settings'] = {'threshold': '0.5'}
            with open(self.config_file, 'w') as f:
                config.write(f)
        else:
            config.read(self.config_file)
        return config

    def __del__(self):
        """Cleanup GPU resources when server is destroyed"""
        try:
            if hasattr(self, 'gpu_resources'):
                print("[CLEANUP] Releasing GPU resources...")
                for resource in self.gpu_resources:
                    try:
                        del resource
                    except:
                        pass
                del self.gpu_resources
            
            if hasattr(self, 'faiss_index'):
                try:
                    del self.faiss_index
                except:
                    pass
            
            # Close thread pools
            if hasattr(self, 'recognition_pool'):
                self.recognition_pool.shutdown(wait=False)
            if hasattr(self, 'gpu_pool'):
                self.gpu_pool.shutdown(wait=False)
            if hasattr(self, 'batch_processing_pool'):
                self.batch_processing_pool.shutdown(wait=False)
            if hasattr(self, 'http_pool'):
                self.http_pool.shutdown(wait=False)
            if hasattr(self, 'mega_batch_pool'):
                self.mega_batch_pool.shutdown(wait=False)
                
        except Exception as e:
            print(f"[ERROR] Cleanup error: {e}")

    def save_embeddings(self):
        """Save embeddings and rebuild indices with proper cleanup"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump({'embeddings': self.face_db}, f)
            print(f"Saved {len(self.face_db)} embeddings to database.")
            
            # Rebuild indices asynchronously with proper cleanup
            def rebuild_task():
                try:
                    # Clean up existing FAISS resources first
                    if hasattr(self, 'gpu_resources'):
                        for resource in self.gpu_resources:
                            try:
                                del resource
                            except:
                                pass
                        del self.gpu_resources
                    
                    if hasattr(self, 'faiss_index'):
                        try:
                            del self.faiss_index
                        except:
                            pass
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Wait for GPU memory to be released
                    import time
                    time.sleep(1)
                    
                    # Rebuild the index
                    self.initialize_faiss_index()
                    
                    if self.gpu_available:
                        self.preload_embeddings_to_gpu()
                        
                except Exception as e:
                    print(f"[ERROR] Failed to rebuild indices: {e}")
            
            self.recognition_pool.submit(rebuild_task)
            
        except Exception as e:
            print(f"[ERROR] Failed to save embeddings: {e}")

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    print(f"Loaded {len(data['embeddings'])} embeddings.")
                    return data['embeddings']
            except Exception as e:
                print(f"[ERROR] Failed to load embeddings: {e}")
                return {}
        return {}

    def get_recognition_result(self, request_id, timeout=30):
        """Get recognition result from cache"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.response_cache:
                result, _ = self.response_cache.pop(request_id)
                return result
            time.sleep(0.01)
        
        return {"status": "error", "message": "Request processing timed out"}

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
        """Get comprehensive performance metrics"""
        uptime = time.time() - self.start_time
        rps = self.processed_count / uptime if uptime > 0 else 0
        avg_processing_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        
        # Calculate cache statistics
        total_l1 = self.cache_stats['l1_hits'] + self.cache_stats['l1_misses']
        total_l2 = self.cache_stats['l2_hits'] + self.cache_stats['l2_misses']
        total_l3 = self.cache_stats['l3_hits'] + self.cache_stats['l3_misses']
        
        return {
            "uptime_seconds": uptime,
            "processed_requests": self.processed_count,
            "requests_per_second": rps,
            "avg_processing_time_ms": avg_processing_time * 1000,
            
            # Queue sizes
            "ultra_priority_queue_size": self.ultra_priority_queue.qsize(),
            "high_priority_queue_size": self.high_priority_queue.qsize(),
            "normal_priority_queue_size": self.normal_priority_queue.qsize(),
            "batch_queue_size": self.batch_request_queue.qsize(),
            "mega_batch_queue_size": self.mega_batch_queue.qsize(),
            "http_broadcast_queue_size": self.http_broadcast_queue.qsize(),
            
            # Database info
            "face_db_size": len(self.face_db),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "faiss_available": self.faiss_available,
            
            # Cache statistics
            "l1_cache_size": len(self.l1_cache),
            "l2_cache_size": len(self.l2_cache),
            "l1_hit_rate": (self.cache_stats['l1_hits'] / total_l1 * 100) if total_l1 > 0 else 0,
            "l2_hit_rate": (self.cache_stats['l2_hits'] / total_l2 * 100) if total_l2 > 0 else 0,
            "l3_hit_rate": (self.cache_stats['l3_hits'] / total_l3 * 100) if total_l3 > 0 else 0,
            
            # System info
            "cpu_cores": self.num_cpu_cores,
            "worker_threads": self.num_worker_threads,
            "gpu_batch_size": self.gpu_batch_size,
            "redis_enabled": self.redis_enabled,
            "gpu_available": self.gpu_available,
            
            "cache_stats": self.cache_stats,
            "recognition_events_count": len(self.recognition_events),
            "response_cache_size": len(self.response_cache),
            "batch_response_cache_size": len(self.batch_response_cache)
        }

    # Legacy method compatibility
    def recognize_face(self, image_data, camera_id=None):
        """Legacy compatibility method"""
        return self.ultra_fast_recognize_face(image_data, camera_id)


# Initialize ultra-fast server with error handling
try:
    face_server = UltraFastFaceRecognitionServer()
    print("[SUCCESS] Ultra-fast face recognition server initialized")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to initialize server: {e}")
    raise

# FLASK ROUTES WITH CORS AND ERROR HANDLING

@app.route('/api/heartbeat', methods=['GET'])
def heartbeat():
    """Ultra-lightweight heartbeat endpoint"""
    try:
        uptime = time.time() - face_server.start_time
        rps = face_server.processed_count / uptime if uptime > 0 else 0
        
        return jsonify({
            "status": "alive", 
            "timestamp": datetime.now().isoformat(),
            "rps": round(rps, 2),
            "ultra_fast": True,
            "processed_requests": face_server.processed_count
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Get comprehensive performance metrics"""
    try:
        return jsonify(face_server.get_performance_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Get detailed cache statistics"""
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
            "l3_cache": {
                "enabled": face_server.redis_enabled,
                "hits": face_server.cache_stats['l3_hits'],
                "misses": face_server.cache_stats['l3_misses']
            },
            "faiss_hits": face_server.cache_stats['faiss_hits'],
            "gpu_hits": face_server.cache_stats['gpu_hits'],
            "system_info": {
                "faiss_available": face_server.faiss_available,
                "gpu_available": face_server.gpu_available,
                "redis_available": face_server.redis_enabled
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_all_caches():
    """Clear all cache levels"""
    try:
        l1_count = len(face_server.l1_cache)
        l2_count = len(face_server.l2_cache)
        
        face_server.l1_cache.clear()
        face_server.l2_cache.clear()
        
        # Clear Redis cache if enabled
        redis_cleared = False
        if face_server.redis_enabled and face_server.redis_client:
            try:
                face_server.redis_client.flushdb()
                redis_cleared = True
            except Exception as e:
                print(f"[WARNING] Failed to clear Redis cache: {e}")
        
        # Reset cache statistics
        face_server.cache_stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0,
            'gpu_hits': 0, 'faiss_hits': 0
        }
        
        return jsonify({
            "status": "success",
            "message": f"Cleared L1 ({l1_count}) and L2 ({l2_count}) caches",
            "redis_cleared": redis_cleared
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize', methods=['POST', 'OPTIONS'])
def recognize():
    """Ultra-fast face recognition with intelligent priority queuing"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        request_id = str(uuid.uuid4())
        camera_id = request.json.get('camera_id')
        priority = request.json.get('priority', 'normal')
        
        current_time = time.time()
        
        # Intelligent priority assignment based on system load
        total_queue_size = (face_server.ultra_priority_queue.qsize() + 
                           face_server.high_priority_queue.qsize() + 
                           face_server.normal_priority_queue.qsize())
        
        if priority == 'ultra' or total_queue_size < 10:
            queue_obj = face_server.ultra_priority_queue
            timeout = 5
        elif priority == 'high' or total_queue_size < 50:
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
    """Ultra-fast batch processing with automatic scaling"""
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
        
        # Intelligent queue selection based on batch size
        if len(files_to_process) >= 500:
            # Mega batch processing
            try:
                face_server.mega_batch_queue.put_nowait((priority, (request_id, files_to_process)))
                processing_mode = "mega_batch"
            except queue.Full:
                return jsonify({"error": "Mega batch processing queue full"}), 503
        else:
            # Regular batch processing
            try:
                face_server.batch_request_queue.put_nowait((priority, (request_id, files_to_process)))
                processing_mode = "batch"
            except queue.Full:
                return jsonify({"error": "Batch processing queue full"}), 503
        
        # Get results
        results = face_server.get_batch_result(request_id)
        
        return jsonify({
            "status": "success",
            "results": results,
            "total_files": len(files_to_process),
            "processing_mode": processing_mode
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reload_embeddings', methods=['POST'])
def reload_embeddings():
    """Reload face embeddings and rebuild indices"""
    try:
        face_server.face_db = face_server.load_embeddings()
        
        # Rebuild indices asynchronously
        def rebuild_task():
            face_server.initialize_faiss_index()
            if face_server.gpu_available:
                face_server.preload_embeddings_to_gpu()
        
        face_server.recognition_pool.submit(rebuild_task)
        
        return jsonify({
            "status": "success", 
            "message": "Embeddings reloaded and indices rebuilding",
            "embedding_count": len(face_server.face_db)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        else:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid threshold value"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_image_to_database', methods=['POST'])
def add_image_to_database():
    """Add a new face image to the database with ultra-fast processing"""
    try:
        if 'image' not in request.json or 'identity' not in request.json:
            return jsonify({"error": "Missing 'image' or 'identity' in request"}), 400

        image_data = request.json['image']
        identity = request.json['identity']

        os.makedirs("Database/U_images", exist_ok=True)

        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Get face analyzer from pool
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

        # Generate timestamps and filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"{identity}_{timestamp}.jpg"
        original_filepath = os.path.join("Database/U_images", original_filename)
        
        # Save image asynchronously
        def save_and_update():
            try:
                cv2.imwrite(original_filepath, img)
                
                # Add to face_db
                face_server.face_db[identity] = {
                    'embedding': embedding,
                    'image_path': original_filepath,
                    'timestamp': timestamp
                }
                
                # Save embeddings and rebuild indices
                face_server.save_embeddings()
                
            except Exception as e:
                print(f"[ERROR] Failed to save image and update database: {e}")
        
        face_server.recognition_pool.submit(save_and_update)

        return jsonify({
            "status": "success",
            "message": f"Image for '{identity}' added to database",
            "filename": original_filename,
            "timestamp": timestamp,
            "ultra_fast_processing": True
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

@app.route('/add_image_to_database_2', methods=['POST'])
def add_image_to_database_2():
    """Add a new face image with multiple face handling and ultra-fast processing"""
    try:
        if 'image' not in request.json or 'identity' not in request.json:
            return jsonify({"error": "Missing 'image' or 'identity' in request"}), 400

        image_data = request.json['image']
        identity = request.json['identity']

        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Get face analyzer from pool
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

        # Select best face if multiple detected
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

        # Generate timestamps and filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"{identity}.jpg"
        original_filepath = os.path.join("Database/images", original_filename)
        
        # Crop image if multiple faces detected
        if multiple_faces_detected:
            x1, y1, x2, y2 = map(int, selected_face.bbox)
            padding_x = int((x2 - x1) * 0.2)
            padding_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(img.shape[1], x2 + padding_x)
            y2 = min(img.shape[0], y2 + padding_y)
            img = img[y1:y2, x1:x2]

        # Save image and update database asynchronously
        def save_and_update():
            try:
                cv2.imwrite(original_filepath, img)
                
                # Add to face_db
                face_server.face_db[identity] = {
                    'embedding': embedding,
                    'image_path': original_filepath,
                    'timestamp': timestamp
                }
                
                # Save embeddings and rebuild indices
                face_server.save_embeddings()
                
            except Exception as e:
                print(f"[ERROR] Failed to save image and update database: {e}")
        
        face_server.recognition_pool.submit(save_and_update)

        response_data = {
            "status": "success",
            "message": f"Image for '{identity}' added to database",
            "filename": original_filename,
            "timestamp": timestamp,
            "ultra_fast_processing": True
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
    """Ultra-fast top matches recognition using FAISS"""
    try:
        if 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        # Convert base64 to image
        nparr = np.frombuffer(base64.b64decode(request.json['image']), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image data"})

        # Get face analyzer from pool
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
        
        # Use FAISS for ultra-fast top-k search
        top_k = request.json.get('top_k', 3)
        min_threshold = request.json.get('min_threshold', 0.2)
        
        search_results = face_server.ultra_fast_similarity_search(query_embedding, top_k=top_k * 2)
        
        # Filter and format results
        top_matches = []
        for identity, similarity in search_results:
            if similarity >= min_threshold and len(top_matches) < top_k:
                # Get image path
                image_path = face_server.face_db.get(identity, {}).get('image_path', '')
                
                matched_image_base64 = None
                if image_path and os.path.exists(image_path):
                    try:
                        matched_image = cv2.imread(image_path)
                        if matched_image is not None:
                            _, buffer = cv2.imencode('.jpg', matched_image)
                            matched_image_base64 = base64.b64encode(buffer).decode('utf-8')
                    except Exception as e:
                        print(f"[ERROR] Failed to encode matched image: {e}")
                
                # Remove @X suffix for consistent identity
                base_identity = identity.split('@')[0]
                
                top_matches.append({
                    "identity": base_identity,
                    "confidence": float(similarity),
                    "matched_image": matched_image_base64
                })

        return jsonify({
            "status": "success",
            "matches": top_matches,
            "ultra_fast_search": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recognition_events', methods=['GET', 'OPTIONS'])
def get_recognition_events():
    """Get recent recognition events with pagination"""
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
            "ultra_fast": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/broadcast_endpoints', methods=['GET'])
def get_broadcast_endpoints():
    """Get configured HTTP broadcast endpoints"""
    try:
        return jsonify({
            "status": "success",
            "endpoints": face_server.http_broadcast_endpoints,
            "ultra_fast": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/broadcast_endpoints', methods=['POST'])
def update_broadcast_endpoints():
    """Update HTTP broadcast endpoints"""
    try:
        data = request.get_json()
        if 'endpoints' not in data or not isinstance(data['endpoints'], list):
            return jsonify({"error": "Invalid endpoints format. Expected a list."}), 400
        
        face_server.http_broadcast_endpoints = data['endpoints']
        
        return jsonify({
            "status": "success",
            "message": "HTTP broadcast endpoints updated",
            "endpoints": face_server.http_broadcast_endpoints,
            "ultra_fast": True
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

# Static file routes
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
    print(f"🚀 Starting ULTRA-FAST Face Recognition Server...")
    print(f"⚡ {face_server.num_worker_threads} workers | {face_server.gpu_batch_size} GPU batch size")
    print(f"💾 {len(face_server.face_db)} face embeddings loaded")
    print(f"🔍 FAISS index: {face_server.faiss_index.ntotal if face_server.faiss_index else 0} entries")
    print(f"🚀 Multi-level caching: L1({face_server.l1_cache_size}) + L2({face_server.l2_cache_size}) + L3(Redis: {face_server.redis_enabled})")
    print(f"📡 HTTP broadcast endpoints: {len(face_server.http_broadcast_endpoints)}")
    print(f"🎯 Target: 100x performance improvement with stability!")
    
    # Use Flask's built-in server with threading
    app.run(host='0.0.0.0', port=3005, debug=False, threaded=True)