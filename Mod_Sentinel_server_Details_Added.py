from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import base64
from datetime import datetime, timedelta
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
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 1. SUPPRESS WARNINGS
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning)

# 2. GPU IMPORTS AND INITIALIZATION
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
    sys.exit(1)

# 3. FAISS INITIALIZATION
try:
    import faiss
    FAISS_AVAILABLE = True
    print(f"[FAISS] Available with {faiss.get_num_gpus()} GPU(s)")
except ImportError:
    print("[ERROR] FAISS not available")
    FAISS_AVAILABLE = False
    sys.exit(1)

# 4. REDIS INITIALIZATION (OPTIONAL)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[WARNING] Redis not available - using local cache only")

app = Flask(__name__)
CORS(app)

# 5. SMART CACHE CLASS FOR PERSON MATCHING
class SmartPersonCache:
    def __init__(self, max_size=10000, duration_hours=1):
        """
        Smart cache that stores matched persons with their embeddings and details
        """
        self.max_size = max_size
        self.duration = timedelta(hours=duration_hours)
        self.cache = OrderedDict()  # {cache_key: cache_data}
        self.cache_lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_stores': 0,
            'cache_evictions': 0
        }
        
        print(f"[CACHE] Initialized Smart Person Cache - Size: {max_size}, Duration: {duration_hours}h")
    
    def _generate_cache_key(self, embedding):
        """Generate unique cache key from embedding"""
        # Use first 64 bytes of embedding for fast hashing
        embedding_bytes = embedding.astype(np.float32).tobytes()[:64]
        return hash(embedding_bytes)
    
    def _is_similar_embedding(self, embedding1, embedding2, threshold=0.5):
        """Check if two embeddings are similar above threshold"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return False, 0.0
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            # Calculate cosine similarity
            similarity = float(np.dot(embedding1_norm, embedding2_norm))
            return similarity >= threshold, similarity
            
        except Exception as e:
            print(f"[CACHE] Similarity calculation error: {e}")
            return False, 0.0
    
    def search_in_cache(self, query_embedding, threshold=0.5):
        """
        Search for similar person in cache
        Returns: (found, person_data, similarity, cache_hit_type)
        """
        with self.cache_lock:
            try:
                current_time = datetime.now()
                expired_keys = []
                best_match = None
                best_similarity = -1
                best_key = None
                
                # Search through cache for similar embeddings
                for cache_key, cache_data in self.cache.items():
                    # Check if expired
                    if current_time - cache_data['timestamp'] > self.duration:
                        expired_keys.append(cache_key)
                        continue
                    
                    # Check similarity with all cached embeddings for this person
                    for cached_embedding in cache_data['embeddings']:
                        is_similar, similarity = self._is_similar_embedding(
                            query_embedding, cached_embedding, threshold
                        )
                        
                        if is_similar and similarity > best_similarity:
                            best_similarity = similarity
                            best_match = cache_data
                            best_key = cache_key
                
                # Clean expired entries
                for key in expired_keys:
                    if key in self.cache:
                        del self.cache[key]
                        self.stats['cache_evictions'] += 1
                
                if best_match:
                    # Move to end (most recently used)
                    self.cache.move_to_end(best_key)
                    self.stats['cache_hits'] += 1
                    
                    # Create response with cache hit info
                    result = best_match['person_data'].copy()
                    result.update({
                        'cache_hit': True,
                        'cache_similarity': best_similarity,
                        'cached_embeddings_count': len(best_match['embeddings']),
                        'cache_age_seconds': (current_time - best_match['timestamp']).total_seconds()
                    })
                    
                    return True, result, best_similarity, 'cache_hit'
                else:
                    self.stats['cache_misses'] += 1
                    return False, None, 0.0, 'cache_miss'
                    
            except Exception as e:
                print(f"[CACHE] Search error: {e}")
                self.stats['cache_misses'] += 1
                return False, None, 0.0, 'cache_error'
    
    def store_in_cache(self, query_embedding, person_data, similarity):
        """
        Store matched person in cache with their embedding
        """
        if similarity < 0.5:  # Only store if similarity >= 50%
            return False
            
        with self.cache_lock:
            try:
                cache_key = self._generate_cache_key(query_embedding)
                current_time = datetime.now()
                
                if cache_key in self.cache:
                    # Update existing entry - add new embedding
                    cache_entry = self.cache[cache_key]
                    cache_entry['embeddings'].append(query_embedding.copy())
                    cache_entry['timestamp'] = current_time
                    cache_entry['access_count'] += 1
                    
                    # Limit embeddings per person to prevent memory bloat
                    if len(cache_entry['embeddings']) > 50:
                        cache_entry['embeddings'] = cache_entry['embeddings'][-25:]  # Keep latest 25
                    
                    # Move to end
                    self.cache.move_to_end(cache_key)
                    
                else:
                    # Create new entry
                    cache_entry = {
                        'embeddings': [query_embedding.copy()],
                        'person_data': person_data.copy(),
                        'timestamp': current_time,
                        'access_count': 1,
                        'original_similarity': similarity
                    }
                    
                    # Add to cache
                    self.cache[cache_key] = cache_entry
                    self.stats['cache_stores'] += 1
                    
                    # Evict oldest if cache full
                    while len(self.cache) > self.max_size:
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                        self.stats['cache_evictions'] += 1
                
                return True
                
            except Exception as e:
                print(f"[CACHE] Store error: {e}")
                return False
    
    def get_stats(self):
        """Get cache statistics"""
        with self.cache_lock:
            total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': round(hit_rate, 2),
                'total_embeddings': sum(len(entry['embeddings']) for entry in self.cache.values()),
                **self.stats
            }
    
    def clear_cache(self):
        """Clear all cache"""
        with self.cache_lock:
            cleared_count = len(self.cache)
            self.cache.clear()
            self.stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'cache_stores': 0,
                'cache_evictions': 0
            }
            return cleared_count

class DynamicParquetReader:
    def __init__(self, parquet_file_path='person_details.parquet'):
        """
        Dynamic parquet reader that monitors file changes and reloads data
        """
        self.parquet_file_path = parquet_file_path
        self.person_details = {}
        self.file_lock = threading.RLock()
        self.last_modified = None
        
        # Load initial data
        self.load_parquet_data()
        
        # Setup file monitoring
        self.setup_file_monitoring()
        
        print(f"[PARQUET] Dynamic reader initialized - File: {parquet_file_path}")
    
    def load_parquet_data(self):
        """Load parquet data into memory"""
        try:
            if not os.path.exists(self.parquet_file_path):
                print(f"[PARQUET] File not found: {self.parquet_file_path}")
                return
            
            # Check if file was modified
            current_modified = os.path.getmtime(self.parquet_file_path)
            if self.last_modified and current_modified == self.last_modified:
                return  # No changes
            
            with self.file_lock:
                # Read parquet file
                df = pd.read_parquet(self.parquet_file_path)
                
                # Validate required columns
                required_columns = ['MID', 'Name', 'Ranking', 'Type', 'Priority']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"[PARQUET] Missing columns: {missing_columns}")
                    return
                
                # Convert to dictionary for fast lookup
                self.person_details = {}
                for _, row in df.iterrows():
                    mid = str(row['MID']).strip()  # Ensure string and remove whitespace
                    self.person_details[mid] = {
                        'name': str(row['Name']).strip(),
                        'ranking': str(row['Ranking']).strip(),
                        'type': str(row['Type']).strip(),
                        'priority': str(row['Priority']).strip(),
                        'mid': mid
                    }
                
                self.last_modified = current_modified
                print(f"[PARQUET] Loaded {len(self.person_details)} person records")
                
        except Exception as e:
            print(f"[PARQUET] Error loading data: {e}")
    
    def get_person_details(self, mid):
        """Get person details by MID"""
        try:
            with self.file_lock:
                # Clean the MID for lookup
                clean_mid = str(mid).strip()
                
                # Direct lookup
                if clean_mid in self.person_details:
                    return self.person_details[clean_mid]
                
                # Try case-insensitive lookup
                for stored_mid, details in self.person_details.items():
                    if stored_mid.lower() == clean_mid.lower():
                        return details
                
                # Try partial matching (in case identity has extra characters)
                base_mid = clean_mid.split('@')[0] if '@' in clean_mid else clean_mid
                for stored_mid, details in self.person_details.items():
                    if stored_mid.lower() == base_mid.lower():
                        return details
                
                return None
                
        except Exception as e:
            print(f"[PARQUET] Error getting person details: {e}")
            return None
    
    def setup_file_monitoring(self):
        """Setup file system monitoring for automatic reloading"""
        try:
            class ParquetFileHandler(FileSystemEventHandler):
                def __init__(self, parquet_reader):
                    self.parquet_reader = parquet_reader
                
                def on_modified(self, event):
                    if not event.is_directory and event.src_path.endswith('.parquet'):
                        if os.path.basename(event.src_path) == os.path.basename(self.parquet_reader.parquet_file_path):
                            print(f"[PARQUET] File changed, reloading: {event.src_path}")
                            # Add small delay to ensure file write is complete
                            threading.Timer(1.0, self.parquet_reader.load_parquet_data).start()
            
            # Start file monitoring
            self.observer = Observer()
            handler = ParquetFileHandler(self)
            
            # Monitor the directory containing the parquet file
            watch_directory = os.path.dirname(os.path.abspath(self.parquet_file_path))
            if not watch_directory:
                watch_directory = '.'
            
            self.observer.schedule(handler, watch_directory, recursive=False)
            self.observer.start()
            
            print(f"[PARQUET] File monitoring started for: {watch_directory}")
            
        except Exception as e:
            print(f"[PARQUET] File monitoring setup failed: {e}")
    
    def stop_monitoring(self):
        """Stop file monitoring - Enhanced error handling"""
        try:
            if hasattr(self, 'observer') and self.observer:
                self.observer.stop()
                # Wait for observer to stop properly
                try:
                    self.observer.join(timeout=5)  # 5 second timeout
                except:
                    # Fallback for older versions
                    self.observer.join()
                print("[PARQUET] File monitoring stopped")
        except Exception as e:
            print(f"[PARQUET] Error stopping monitoring: {e}")
    
    def get_stats(self):
        """Get parquet reader statistics"""
        with self.file_lock:
            return {
                'total_records': len(self.person_details),
                'file_path': self.parquet_file_path,
                'last_modified': self.last_modified,
                'file_exists': os.path.exists(self.parquet_file_path)
            }
    
    def _save_parquet_data(self):
        """Save current person details to parquet file"""
        try:
            # Convert person_details dict to DataFrame
            data_for_df = []
            for mid, details in self.person_details.items():
                data_for_df.append({
                    'MID': details['mid'],
                    'Name': details['name'],
                    'Ranking': details['ranking'],
                    'Type': details['type'],
                    'Priority': details['priority']
                })
            
            if data_for_df:
                df = pd.DataFrame(data_for_df)
            else:
                # Create empty DataFrame with proper columns
                df = pd.DataFrame(columns=['MID', 'Name', 'Ranking', 'Type', 'Priority'])
            
            # Save to parquet
            df.to_parquet(self.parquet_file_path, index=False)
            
            # Update last modified time
            self.last_modified = os.path.getmtime(self.parquet_file_path)
            
            print(f"[PARQUET] Saved {len(data_for_df)} records to {self.parquet_file_path}")
            return True
            
        except Exception as e:
            print(f"[PARQUET] Error saving data: {e}")
            return False
    
    def add_person_details(self, mid, name, ranking, type_val, priority):
        """Add new person details"""
        try:
            with self.file_lock:
                clean_mid = str(mid).strip()
                
                # Check if already exists
                if clean_mid in self.person_details:
                    return {
                        'success': False,
                        'error': f"Person with MID '{clean_mid}' already exists"
                    }
                
                # Add to memory
                self.person_details[clean_mid] = {
                    'mid': clean_mid,
                    'name': str(name).strip(),
                    'ranking': str(ranking).strip(),
                    'type': str(type_val).strip(),
                    'priority': str(priority).strip()
                }
                
                # Save to file
                if self._save_parquet_data():
                    return {
                        'success': True,
                        'total_records': len(self.person_details)
                    }
                else:
                    # Rollback memory change if save failed
                    if clean_mid in self.person_details:
                        del self.person_details[clean_mid]
                    return {
                        'success': False,
                        'error': 'Failed to save to parquet file'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_person_details(self, mid, name, ranking, type_val, priority):
        """Update existing person details"""
        try:
            with self.file_lock:
                clean_mid = str(mid).strip()
                
                # Check if exists
                if clean_mid not in self.person_details:
                    return {
                        'success': False,
                        'error': f"Person with MID '{clean_mid}' not found"
                    }
                
                # Backup current data
                backup_data = self.person_details[clean_mid].copy()
                
                # Update in memory
                self.person_details[clean_mid] = {
                    'mid': clean_mid,
                    'name': str(name).strip(),
                    'ranking': str(ranking).strip(),
                    'type': str(type_val).strip(),
                    'priority': str(priority).strip()
                }
                
                # Save to file
                if self._save_parquet_data():
                    return {
                        'success': True,
                        'total_records': len(self.person_details)
                    }
                else:
                    # Rollback to backup if save failed
                    self.person_details[clean_mid] = backup_data
                    return {
                        'success': False,
                        'error': 'Failed to save to parquet file'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_person_details(self, mid):
        """Delete person details"""
        try:
            with self.file_lock:
                clean_mid = str(mid).strip()
                
                # Check if exists
                if clean_mid not in self.person_details:
                    return {
                        'success': False,
                        'error': f"Person with MID '{clean_mid}' not found"
                    }
                
                # Backup data for rollback
                backup_data = self.person_details[clean_mid].copy()
                
                # Delete from memory
                del self.person_details[clean_mid]
                
                # Save to file
                if self._save_parquet_data():
                    return {
                        'success': True,
                        'total_records': len(self.person_details)
                    }
                else:
                    # Rollback if save failed
                    self.person_details[clean_mid] = backup_data
                    return {
                        'success': False,
                        'error': 'Failed to save to parquet file'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def bulk_add_persons(self, persons_list):
        """Bulk add multiple person details"""
        try:
            with self.file_lock:
                # Backup current data for rollback
                backup_data = self.person_details.copy()
                
                # Add all persons to memory
                added_count = 0
                for person in persons_list:
                    clean_mid = str(person['mid']).strip()
                    
                    # Skip if already exists
                    if clean_mid in self.person_details:
                        continue
                    
                    self.person_details[clean_mid] = {
                        'mid': clean_mid,
                        'name': str(person['name']).strip(),
                        'ranking': str(person['ranking']).strip(),
                        'type': str(person['type']).strip(),
                        'priority': str(person['priority']).strip()
                    }
                    added_count += 1
                
                if added_count == 0:
                    return {
                        'success': False,
                        'error': 'No new records to add (all MIDs already exist)'
                    }
                
                # Save to file
                if self._save_parquet_data():
                    return {
                        'success': True,
                        'added_count': added_count,
                        'total_records': len(self.person_details)
                    }
                else:
                    # Rollback to backup if save failed
                    self.person_details = backup_data
                    return {
                        'success': False,
                        'error': 'Failed to save to parquet file'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def bulk_delete_persons(self, mids_list):
        """Bulk delete multiple person details"""
        try:
            with self.file_lock:
                # Backup current data for rollback
                backup_data = self.person_details.copy()
                
                # Delete from memory
                deleted_count = 0
                for mid in mids_list:
                    clean_mid = str(mid).strip()
                    if clean_mid in self.person_details:
                        del self.person_details[clean_mid]
                        deleted_count += 1
                
                if deleted_count == 0:
                    return {
                        'success': False,
                        'error': 'No records found to delete'
                    }
                
                # Save to file
                if self._save_parquet_data():
                    return {
                        'success': True,
                        'deleted_count': deleted_count,
                        'total_records': len(self.person_details)
                    }
                else:
                    # Rollback to backup if save failed
                    self.person_details = backup_data
                    return {
                        'success': False,
                        'error': 'Failed to save to parquet file'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# 6. MAIN FACE RECOGNITION SERVER CLASS
class OptimizedFaceRecognitionServer:
    def __init__(self):
        print("[INIT] Starting Optimized Face Recognition Server with Smart Caching...")
        
        # 6.1 CORE CONFIGURATION
        self.config_file = 'config.ini'
        self.config = self.load_config()
        self.threshold = float(self.config.get('Settings', 'threshold', fallback=0.5))
        self.cache_threshold = 0.5  # 50% threshold for caching
        
        # 6.2 GPU SETUP
        self.gpu_memory_limit = 16 * 1024 * 1024 * 1024  # 16GB
        self.gpu_device_id = 0
        self.setup_gpu_environment()
        
        # 6.3 PERFORMANCE CONFIGURATION
        self.num_cpu_cores = os.cpu_count()
        self.num_worker_threads = min(24, self.num_cpu_cores * 0.75)
        self.gpu_batch_size = 256
        self.embedding_batch_size = 2048
        
        # 6.4 INITIALIZE SMART CACHE
        self.smart_cache = SmartPersonCache(max_size=10000, duration_hours=1)

        # 6.4.1 INITIALIZE DYNAMIC PARQUET READER
        self.parquet_reader = DynamicParquetReader('person_details.parquet')
        
        # 6.5 GPU MEMORY POOL
        self.initialize_gpu_memory_pool()
        
        # 6.6 FACE ANALYZER POOL
        self.face_analyzer_pool = queue.Queue(maxsize=16)
        self.face_analyzer_lock = threading.Lock()
        self.initialize_face_analyzer_pool()
        
        # 6.7 DATABASE AND INDEXING
        self.embeddings_file = 'Database/face_embeddings.pkl'
        self.face_db = self.load_embeddings()
        
        # 6.8 FAISS GPU INDEX
        self.faiss_index = None
        self.embedding_id_mapping = {}
        self.faiss_lock = threading.RLock()
        self.initialize_robust_faiss_index()
        
        # 6.9 PROCESSING QUEUES (SINGLE HIGH PRIORITY)
        self.recognition_queue = queue.Queue(maxsize=2000)
        self.batch_request_queue = queue.Queue(maxsize=200)
        
        # 6.10 RESPONSE CACHES
        self.response_cache = {}
        self.batch_response_cache = {}
        self.recognition_events = deque(maxlen=20000)
        self.http_broadcast_queue = queue.Queue(maxsize=2000)
        
        # 6.11 PERFORMANCE METRICS
        self.processed_count = 0
        self.start_time = time.time()
        self.request_times = deque(maxlen=10000)
        self.gpu_memory_usage = deque(maxlen=2000)
        
        # 6.12 THREAD POOLS
        self.recognition_pool = ThreadPoolExecutor(max_workers=self.num_worker_threads, thread_name_prefix="Recognition")
        self.http_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="HTTP")
        self.batch_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="Batch")
        
        # 6.13 DIRECTORY SETUP
        for directory in ["Get/In", "Get/Out", "static/images", "static/varient", "Logs", "Database/images", "Database/U_images"]:
            os.makedirs(directory, exist_ok=True)
        
        # 6.14 API ENDPOINTS
        self.log_api_endpoint = "http://192.168.14.102:7578/api/FaceRecognition/Recognize-Logs"
        self.log_api_endpoint_2 = "http://192.168.15.129:5002/add_log"

        
        # 6.15 START ALL WORKERS
        self.start_all_workers()
        
        # 6.16 SETUP GRACEFUL SHUTDOWN
        self.setup_graceful_shutdown()
        
        # 6.17 GPU MONITORING
        self.start_gpu_monitoring()
        
        print(f"[SUCCESS] Server initialized with Smart Caching:")
        print(f"  - GPU Memory Reserved: {self.gpu_memory_limit / (1024**3):.1f}GB")
        print(f"  - Worker Threads: {self.num_worker_threads}")
        print(f"  - Face Analyzers: {self.face_analyzer_pool.qsize()}")
        print(f"  - FAISS Index: {self.faiss_index.ntotal if self.faiss_index else 0} embeddings")
        print(f"  - Smart Cache: {self.smart_cache.max_size} entries, {self.smart_cache.duration}")

    # 7. GPU ENVIRONMENT SETUP
    def setup_gpu_environment(self):
        """Setup GPU environment with proper memory management"""
        try:
            torch.cuda.set_device(self.gpu_device_id)
            
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_device_id)
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['CUDA_CACHE_DISABLE'] = '0'
            
            if 'cp' in globals():
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=self.gpu_memory_limit)
                print(f"[GPU] CuPy memory pool limited to {self.gpu_memory_limit / (1024**3):.1f}GB")
            
            torch.cuda.empty_cache()
            
            gpu_name = torch.cuda.get_device_name(self.gpu_device_id)
            gpu_memory = torch.cuda.get_device_properties(self.gpu_device_id).total_memory
            
            print(f"[GPU] Using {gpu_name} with {gpu_memory / (1024**3):.1f}GB total memory")
            
        except Exception as e:
            print(f"[ERROR] GPU setup failed: {e}")
            raise

    # 8. GPU MEMORY POOL INITIALIZATION
    def initialize_gpu_memory_pool(self):
        """Initialize GPU memory pool to prevent fragmentation"""
        try:
            if 'cp' in globals():
                test_size = min(1024 * 1024 * 1024, self.gpu_memory_limit // 4)
                test_array = cp.zeros(test_size // 4, dtype=cp.float32)
                del test_array
                
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=self.gpu_memory_limit)
                
                print(f"[GPU] Memory pool initialized with {self.gpu_memory_limit / (1024**3):.1f}GB limit")
            
        except Exception as e:
            print(f"[WARNING] GPU memory pool initialization failed: {e}")

    # 9. FACE ANALYZER POOL INITIALIZATION
    def initialize_face_analyzer_pool(self):
        """Initialize face analyzer pool with proper GPU context management"""
        print("[INIT] Creating face analyzer pool...")
        
        pool_size = 8
        successful_analyzers = 0
        
        for i in range(pool_size):
            try:
                analyzer = FaceAnalysis(
                    name='buffalo_l',
                    root='models',
                    providers=['CUDAExecutionProvider']
                )
                
                analyzer.prepare(ctx_id=self.gpu_device_id, det_size=(640, 640))
                
                # Test the analyzer
                test_img = np.zeros((224, 224, 3), dtype=np.uint8)
                test_faces = analyzer.get(test_img)
                
                self.face_analyzer_pool.put(analyzer)
                successful_analyzers += 1
                print(f"[INIT] Created face analyzer {successful_analyzers}/{pool_size}")
                
            except Exception as e:
                print(f"[ERROR] Failed to create face analyzer {i}: {e}")
                
        if successful_analyzers == 0:
            raise RuntimeError("Failed to create any face analyzers")
            
        print(f"[SUCCESS] Created {successful_analyzers} face analyzers")

    # 10. FACE ANALYZER MANAGEMENT
    def get_face_analyzer(self):
        """Get face analyzer from pool with timeout"""
        try:
            return self.face_analyzer_pool.get(timeout=30.0)
        except queue.Empty:
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
            pass

    # 11. FAISS INDEX INITIALIZATION
    def initialize_robust_faiss_index(self):
        """Initialize FAISS index with robust GPU memory management"""
        if not FAISS_AVAILABLE:
            print("[ERROR] FAISS not available")
            return
            
        print("[INIT] Building FAISS GPU index...")
        
        if not self.face_db:
            print("[WARNING] No embeddings to index")
            return
        
        with self.faiss_lock:
            try:
                embeddings = []
                identities = []
                
                for identity, data in self.face_db.items():
                    if 'embedding' in data and data['embedding'] is not None:
                        embeddings.append(data['embedding'])
                        identities.append(identity)
                
                if not embeddings:
                    print("[WARNING] No valid embeddings found")
                    return
                
                embeddings_array = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings_array = embeddings_array / norms
                
                dimension = embeddings_array.shape[1]
                print(f"[FAISS] Building index for {len(embeddings_array)} embeddings, dimension {dimension}")
                
                gpu_resources = faiss.StandardGpuResources()
                
                available_memory = self.get_available_gpu_memory()
                faiss_memory_limit = min(
                    2 * 1024 * 1024 * 1024,
                    available_memory // 4
                )
                gpu_resources.setTempMemory(faiss_memory_limit)
                
                print(f"[FAISS] Allocated {faiss_memory_limit / (1024**2):.0f}MB for FAISS operations")
                
                config = faiss.GpuIndexFlatConfig()
                config.device = self.gpu_device_id
                config.useFloat16 = False
                config.usePrecomputed = False
                
                self.faiss_index = faiss.GpuIndexFlatIP(gpu_resources, dimension, config)
                
                batch_size = self.embedding_batch_size
                total_added = 0
                
                for i in range(0, len(embeddings_array), batch_size):
                    batch_end = min(i + batch_size, len(embeddings_array))
                    batch_embeddings = embeddings_array[i:batch_end]
                    
                    self.faiss_index.add(batch_embeddings)
                    total_added += len(batch_embeddings)
                    
                    torch.cuda.synchronize()
                    
                    print(f"[FAISS] Added {total_added}/{len(embeddings_array)} embeddings")
                
                self.embedding_id_mapping = {i: identities[i] for i in range(len(identities))}
                self.gpu_resources = gpu_resources
                
                print(f"[SUCCESS] FAISS GPU index built with {self.faiss_index.ntotal} embeddings")
                
            except Exception as e:
                print(f"[ERROR] Failed to build FAISS index: {e}")
                self.faiss_index = None
                self.cleanup_faiss_resources()

    # 12. GPU MEMORY MANAGEMENT
    def get_available_gpu_memory(self):
        """Get available GPU memory in bytes"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                return torch.cuda.get_device_properties(self.gpu_device_id).total_memory - torch.cuda.memory_allocated(self.gpu_device_id)
            return 0
        except Exception:
            return 2 * 1024 * 1024 * 1024

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

    # 13. FAISS SEARCH METHODS
    def safe_faiss_search(self, query_embedding, top_k=1):
        """Thread-safe FAISS search with robust error handling"""
        if not self.faiss_index:
            return []
        
        with self.faiss_lock:
            try:
                query_embedding = query_embedding.astype(np.float32)
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
                query_embedding = query_embedding.reshape(1, -1)
                
                query_copy = np.copy(query_embedding)
                
                start_time = time.time()
                similarities, indices = self.faiss_index.search(query_copy, top_k)
                search_time = time.time() - start_time
                
                torch.cuda.synchronize()
                
                results = []
                for similarity, index in zip(similarities[0], indices[0]):
                    if index != -1 and index < len(self.embedding_id_mapping):
                        identity = self.embedding_id_mapping[index]
                        results.append((identity, float(similarity)))
                
                if search_time > 0.1:
                    print(f"[WARNING] Slow FAISS search: {search_time:.3f}s")
                
                return results
                
            except Exception as e:
                print(f"[ERROR] FAISS search failed: {e}")
                self.schedule_faiss_recovery()
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
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"[ERROR] Numpy similarity search failed: {e}")
            return []

    def schedule_faiss_recovery(self):
        """Schedule FAISS index recovery in background"""
        def recovery_task():
            try:
                print("[RECOVERY] Starting FAISS index recovery...")
                
                self.cleanup_faiss_resources()
                time.sleep(2)
                self.initialize_robust_faiss_index()
                
                if self.faiss_index:
                    print("[RECOVERY] FAISS index recovered successfully")
                else:
                    print("[RECOVERY] FAISS recovery failed")
                    
            except Exception as e:
                print(f"[ERROR] FAISS recovery failed: {e}")
        
        self.recognition_pool.submit(recovery_task)

    # 14. MAIN FACE RECOGNITION METHOD WITH SMART CACHING
    def smart_recognize_face(self, image_data, camera_id=None):
        """
        Smart face recognition with advanced caching system
        
        Flow:
        1. Extract embedding from image
        2. Check smart cache first (50% threshold)
        3. If cache hit - return cached person data with new confidence
        4. If cache miss - search database
        5. If database match (>= 50%) - store in cache and return
        6. If no match - handle as unrecognized
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_id = str(uuid.uuid4())[:8]
        
        try:
            # 14.1 DECODE IMAGE
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"status": "error", "message": "Invalid image data"}

            # 14.2 SAVE INPUT IMAGE
            prefix = f"cam{camera_id}_{request_id}_" if camera_id else f"{request_id}_"
            input_filename = self.save_image_async(img, "Get/In", f"{prefix}input")

            # 14.3 FACE DETECTION
            face_analyzer = self.get_face_analyzer()
            
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_analyzer.get(img_rgb)
                self.return_face_analyzer(face_analyzer)
            except Exception as e:
                self.return_face_analyzer(face_analyzer)
                raise e

            if len(faces) == 0:
                return self.handle_no_face_detected(input_filename, prefix, camera_id, request_id)

            # 14.4 EXTRACT EMBEDDING
            query_embedding = faces[0].embedding

            # 14.5 SMART CACHE CHECK (50% THRESHOLD)
            print(f"[CACHE] Checking smart cache for request {request_id}")
            cache_found, cache_result, cache_similarity, cache_type = self.smart_cache.search_in_cache(
                query_embedding, threshold=self.cache_threshold
            )
            
            if cache_found:
                print(f"[CACHE] Cache HIT - Similarity: {cache_similarity:.3f} - Type: {cache_type}")
                
                # Update result with current request info
                cache_result.update({
                    "input_filename": input_filename,
                    "request_id": request_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "confidence": cache_similarity,  # Use cache similarity as confidence
                    "status": "matched",
                    "cache_hit": True,
                    "cache_type": cache_type
                })
                
                # Log cache hit
                self.log_recognition_async(cache_result, camera_id)
                
                # Store current embedding in cache (add to existing person's embeddings)
                self.smart_cache.store_in_cache(query_embedding, cache_result, cache_similarity)
                
                return cache_result

            # 14.6 CACHE MISS - SEARCH DATABASE
            print(f"[CACHE] Cache MISS - Searching database for request {request_id}")
            
            # Search using FAISS
            search_results = self.safe_faiss_search(query_embedding, top_k=1)
            
            # Fallback to numpy if FAISS fails
            if not search_results:
                search_results = self.numpy_similarity_search(query_embedding, top_k=1)
            
            if search_results:
                best_identity, best_similarity = search_results[0]
                
                if best_similarity > self.threshold:
                    print(f"[DB] Database match found - Identity: {best_identity} - Similarity: {best_similarity:.3f}")
                    
                    # Create match result
                    result = self.create_match_result(
                        best_identity, best_similarity, input_filename, 
                        prefix, camera_id, request_id, query_embedding
                    )
                    
                    # Store in smart cache if similarity >= 50%
                    if best_similarity >= self.cache_threshold:
                        cache_stored = self.smart_cache.store_in_cache(query_embedding, result, best_similarity)
                        if cache_stored:
                            print(f"[CACHE] Stored new person in cache - Identity: {best_identity}")
                            result['cached'] = True
                        else:
                            result['cached'] = False
                    
                    return result
            
            # 14.7 NO MATCH FOUND
            print(f"[DB] No match found in database for request {request_id}")
            return self.handle_unrecognized_face(
                img, faces[0], input_filename, prefix, camera_id, request_id
            )
            
        except Exception as e:
            print(f"[ERROR] Recognition error: {e}")
            return {
                "status": "error",
                "message": f"Recognition error: {str(e)}",
                "camera_id": camera_id,
                "request_id": request_id
            }

    # 15. HELPER METHODS FOR RESULT CREATION
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
                "request_id": request_id,
                "cache_hit": False
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
        """Create match result with dynamic parquet details"""
        try:
            # Find image path
            image_path = None
            for face_identity, data in self.face_db.items():
                if face_identity.split('@')[0] == identity.split('@')[0]:
                    image_path = data.get('image_path')
                    break
            
            if not image_path or not os.path.exists(image_path):
                matched_image = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.putText(matched_image, "MATCHED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                matched_image = cv2.imread(image_path)
            
            output_filename = self.save_image_async(matched_image, "Get/Out", f"{prefix}match")
            
            _, buffer = cv2.imencode('.jpg', matched_image)
            matched_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Get clean identity (MID)
            clean_identity = identity.split('@')[0]
            
            # Get person details from parquet file
            person_details = self.parquet_reader.get_person_details(clean_identity)
            
            result = {
                "status": "matched",
                "confidence": float(similarity),
                "matched_image": matched_image_base64,
                "input_filename": input_filename,
                "output_filename": output_filename,
                "matched_filename": os.path.basename(image_path) if image_path else "placeholder.jpg",
                "identity": clean_identity,
                "camera_id": camera_id,
                "request_id": request_id,
                "gpu_accelerated": True,
                "cache_hit": False,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add person details if found in parquet
            if person_details:
                result.update({
                    "person_name": person_details['name'],
                    "person_ranking": person_details['ranking'],
                    "person_type": person_details['type'],
                    "person_priority": person_details['priority'],
                    "person_mid": person_details['mid'],
                    "parquet_lookup": "success"
                })
                print(f"[PARQUET] Found details for {clean_identity}: {person_details['name']}")
            else:
                result.update({
                    "person_name": clean_identity,  # Fallback to identity
                    "person_ranking": "Unknown",
                    "person_type": "Unknown", 
                    "person_priority": "Unknown",
                    "person_mid": clean_identity,
                    "parquet_lookup": "not_found"
                })
                print(f"[PARQUET] No details found for {clean_identity}")
            
            # Log asynchronously
            self.log_recognition_async(result, camera_id)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Create match result failed: {e}")
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
                "request_id": request_id,
                "cache_hit": False,
                "timestamp": datetime.now().isoformat()
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

    # 16. LOGGING METHODS
    def log_recognition_async(self, result, camera_id=None):
        """Log recognition details asynchronously with person details"""
        def log_task():
            try:
                event_data = {
                    "timestamp": datetime.now().isoformat(),
                    "status": result.get('status', 'Unknown'),
                    "camera_id": camera_id,
                    "camera_name": self.get_camera_name(camera_id),
                    "confidence": result.get('confidence', 0.0),
                    "matched_filename": result.get('matched_filename', ''),
                    "identity": result.get('identity', ''),
                    "input_image": f"/Get/In/{result.get('input_filename', '')}" if result.get('input_filename') else None,
                    "matched_image": f"Database/images/{result.get('matched_filename', '')}" if result.get('matched_filename') else None,
                    "cache_hit": result.get('cache_hit', False),
                    "cache_type": result.get('cache_type', ''),
                    "request_id": result.get('request_id', ''),
                    "gpu_accelerated": True,
                    "cached": result.get('cached', False),
                    
                    # Add person details from parquet
                    "person_name": result.get('person_name', ''),
                    "person_ranking": result.get('person_ranking', ''),
                    "person_type": result.get('person_type', ''),
                    "person_priority": result.get('person_priority', ''),
                    "person_mid": result.get('person_mid', ''),
                    "parquet_lookup": result.get('parquet_lookup', 'not_attempted')
                }

                self.recognition_events.append(event_data)
                
                try:
                    self.http_broadcast_queue.put_nowait(event_data)
                except queue.Full:
                    pass

                self.send_api_logs_async(result, camera_id)
                    
            except Exception as e:
                print(f"[ERROR] Logging failed: {e}")
        
        self.recognition_pool.submit(log_task)

    def send_api_logs_async(self, result, camera_id):
        """Send API logs asynchronously with person details"""
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
                    "identity": result.get('identity', ''),
                    "cache_hit": result.get('cache_hit', False),
                    "gpu_accelerated": True,
                    
                    # Add person details
                    "person_name": result.get('person_name', ''),
                    "person_ranking": result.get('person_ranking', ''),
                    "person_type": result.get('person_type', ''),
                    "person_priority": result.get('person_priority', ''),
                    "person_mid": result.get('person_mid', ''),
                    "parquet_lookup": result.get('parquet_lookup', 'not_attempted')
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
                        pass
            except Exception as e:
                print(f"[ERROR] API logging failed: {e}")
        
        self.recognition_pool.submit(api_task)

    def get_camera_name(self, camera_id):
        """Get camera name"""
        camera_names = {
            1: 'Paiza Gate 1', 2: 'Paiza Gate 2', 3: 'Paiza Gate 3', 4: 'Paiza Gate 4',
            31: 'Travel Counter Out', 32: 'Travel Counter In',
            61: 'Membership Counter Out', 62: 'Membership Counter In',
            1131: 'Travel Counter Out Sup', 1132: 'Travel Counter In Sup',
            1161: 'Membership Counter Out Sup', 1162: 'Membership Counter In Sup',
            3701: 'Table 37 Left', 3702: 'Table 37 Right'
        }
        return camera_names.get(camera_id, f"Camera {camera_id}")

    # 17. WORKER THREAD INITIALIZATION
    def start_all_workers(self):
        """Start all worker threads"""
        # Recognition workers (high priority only)
        for i in range(self.num_worker_threads):
            threading.Thread(target=self.process_recognition_queue, daemon=True, name=f"Recognition-{i}").start()
        
        # Batch processing workers
        for i in range(2):
            threading.Thread(target=self.process_batch_queue, daemon=True, name=f"Batch-{i}").start()
        
        # HTTP workers
        for i in range(4):
            threading.Thread(target=self.process_http_broadcasts, daemon=True, name=f"HTTP-{i}").start()
        
        # Maintenance workers
        threading.Thread(target=self.cache_maintenance_worker, daemon=True, name="CacheMaintenance").start()
        threading.Thread(target=self.performance_monitor_worker, daemon=True, name="PerfMonitor").start()
        
        print(f"[WORKERS] Started {self.num_worker_threads} recognition workers + maintenance workers")

    def process_recognition_queue(self):
        """Process recognition queue (single high priority)"""
        while True:
            try:
                request_data = self.recognition_queue.get(timeout=0.1)
                request_id, image_data, camera_id = request_data
                
                start_time = time.time()
                result = self.smart_recognize_face(image_data, camera_id)
                processing_time = time.time() - start_time
                
                self.response_cache[request_id] = (result, time.time())
                self.processed_count += 1
                self.request_times.append(processing_time)
                
                self.recognition_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] Recognition queue processing: {e}")
                time.sleep(0.1)

    # 18. BATCH PROCESSING
    def process_batch_queue(self):
        """Process batch queue"""
        while True:
            try:
                request_id, files_to_process = self.batch_request_queue.get()
                
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
            result = self.smart_recognize_face(image_data)
            
            # Format for batch response
            batch_result = {
                "image_id": image_id,
                "status": result.get('status', 'error'),
                "filename": filename,
                "request_id": request_id,
                "cache_hit": result.get('cache_hit', False)
            }
            
            if result.get('status') == 'matched':
                batch_result.update({
                    "confidence": result.get('confidence', 0.0),
                    "identity": result.get('identity', ''),
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

    # 19. HTTP BROADCAST PROCESSING
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

    # 20. MAINTENANCE WORKERS
    def cache_maintenance_worker(self):
        """Maintain caches and clean up"""
        while True:
            try:
                time.sleep(30)
                current_time = time.time()
                
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
                
                # Log cache stats
                if self.processed_count % 100 == 0:
                    cache_stats = self.smart_cache.get_stats()
                    print(f"[CACHE] Hit Rate: {cache_stats['hit_rate']}% | "
                          f"Size: {cache_stats['cache_size']}/{cache_stats['max_size']} | "
                          f"Embeddings: {cache_stats['total_embeddings']}")
                
            except Exception as e:
                print(f"[ERROR] Cache maintenance: {e}")

    def performance_monitor_worker(self):
        """Monitor performance"""
        while True:
            try:
                time.sleep(60)
                
                uptime = time.time() - self.start_time
                rps = self.processed_count / uptime if uptime > 0 else 0
                avg_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
                
                cache_stats = self.smart_cache.get_stats()
                
                print(f"[PERF] RPS: {rps:.1f} | Avg: {avg_time*1000:.1f}ms | "
                      f"Cache Hit: {cache_stats['hit_rate']:.1f}% | "
                      f"Processed: {self.processed_count}")
                
            except Exception as e:
                print(f"[ERROR] Performance monitor: {e}")

    # 21. GPU MONITORING
    def start_gpu_monitoring(self):
        """Start GPU health monitoring"""
        def gpu_monitor():
            while True:
                try:
                    time.sleep(60)
                    
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
                        
                        if usage_percent > 90:
                            print(f"[WARNING] High GPU memory usage: {usage_percent:.1f}%")
                            torch.cuda.empty_cache()
                        
                        if self.processed_count % 100 == 0:
                            print(f"[GPU] Memory: {usage_percent:.1f}% ({memory_allocated/(1024**3):.1f}GB/{memory_total/(1024**3):.1f}GB)")
                    
                except Exception as e:
                    print(f"[ERROR] GPU monitoring: {e}")
        
        threading.Thread(target=gpu_monitor, daemon=True, name="GPUMonitor").start()

    # 22. GRACEFUL SHUTDOWN
    def setup_graceful_shutdown(self):
        """Setup graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"[SHUTDOWN] Received signal {signum}, shutting down gracefully...")
            self.cleanup_and_exit()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.cleanup_and_exit)

    def cleanup_and_exit(self):
        """Cleanup resources and exit - Compatible with older Python versions"""
        try:
            print("[CLEANUP] Starting cleanup...")
            
            # Stop parquet monitoring
            if hasattr(self, 'parquet_reader'):
                self.parquet_reader.stop_monitoring()
            
            # Updated ThreadPoolExecutor shutdown for compatibility
            if hasattr(self, 'recognition_pool'):
                try:
                    self.recognition_pool.shutdown(wait=True)
                    # For older Python versions, wait manually
                    import time
                    time.sleep(2)
                except TypeError:
                    # Fallback for even older versions
                    self.recognition_pool.shutdown()
                    
            if hasattr(self, 'http_pool'):
                try:
                    self.http_pool.shutdown(wait=True)
                    time.sleep(1)
                except TypeError:
                    self.http_pool.shutdown()
                    
            if hasattr(self, 'batch_pool'):
                try:
                    self.batch_pool.shutdown(wait=True)
                    time.sleep(1)
                except TypeError:
                    self.batch_pool.shutdown()
            
            self.cleanup_faiss_resources()
            torch.cuda.empty_cache()
            
            print("[CLEANUP] Cleanup completed")
            
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {e}")


    # 23. DATABASE MANAGEMENT METHODS
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
            
            # Clear smart cache when database changes
            cleared_count = self.smart_cache.clear_cache()
            print(f"[CACHE] Cleared {cleared_count} cache entries due to database update")
            
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

    # 24. RESULT RETRIEVAL METHODS
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

    # 25. PERFORMANCE METRICS
    def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        uptime = time.time() - self.start_time
        rps = self.processed_count / uptime if uptime > 0 else 0
        avg_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        
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
        
        # Smart cache stats
        cache_stats = self.smart_cache.get_stats()
        
        return {
            # Performance metrics
            "uptime_seconds": uptime,
            "processed_requests": self.processed_count,
            "requests_per_second": rps,
            "avg_processing_time_ms": avg_time * 1000,
            
            # Queue sizes
            "recognition_queue_size": self.recognition_queue.qsize(),
            "batch_queue_size": self.batch_request_queue.qsize(),
            "http_broadcast_queue_size": self.http_broadcast_queue.qsize(),
            
            # Database and indexing
            "face_db_size": len(self.face_db),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            
            # Smart cache metrics
            "smart_cache": cache_stats,
            
            # System info
            "gpu_memory": gpu_memory_info,
            "recognition_events_count": len(self.recognition_events),
            "response_cache_size": len(self.response_cache),
            "batch_response_cache_size": len(self.batch_response_cache),
            "threshold": self.threshold,
            "cache_threshold": self.cache_threshold,
            "gpu_accelerated": True,
            "smart_caching_enabled": True
        }

    # Legacy compatibility method
    def recognize_face(self, image_data, camera_id=None):
        """Legacy compatibility method"""
        return self.smart_recognize_face(image_data, camera_id)


# 26. INITIALIZE SERVER
try:
    face_server = OptimizedFaceRecognitionServer()
    print("[SUCCESS] Optimized Face Recognition Server with Smart Caching initialized")
except Exception as e:
    print(f"[CRITICAL] Server initialization failed: {e}")
    sys.exit(1)

# 27. FLASK ROUTES
@app.route('/api/heartbeat', methods=['GET'])
def heartbeat():
    """Heartbeat endpoint with cache stats"""
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
        
        cache_stats = face_server.smart_cache.get_stats()
        
        return jsonify({
            "status": "alive",
            "timestamp": datetime.now().isoformat(),
            "rps": round(rps, 2),
            "processed_requests": face_server.processed_count,
            "gpu_accelerated": True,
            "gpu_info": gpu_info,
            "smart_cache_hit_rate": cache_stats['hit_rate'],
            "smart_cache_size": cache_stats['cache_size'],
            "optimized_caching": True
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
#==========================================================================================================================================================
@app.route('/api/parquet/stats', methods=['GET'])
def parquet_stats():
    """Get parquet reader statistics"""
    try:
        stats = face_server.parquet_reader.get_stats()
        return jsonify({
            "status": "success",
            "parquet_stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parquet/reload', methods=['POST'])
def reload_parquet():
    """Manually reload parquet data"""
    try:
        face_server.parquet_reader.load_parquet_data()
        stats = face_server.parquet_reader.get_stats()
        return jsonify({
            "status": "success",
            "message": "Parquet data reloaded",
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parquet/lookup/<mid>', methods=['GET'])
def lookup_person_details(mid):
    """Lookup person details by MID"""
    try:
        details = face_server.parquet_reader.get_person_details(mid)
        if details:
            return jsonify({
                "status": "success",
                "person_details": details
            })
        else:
            return jsonify({
                "status": "not_found",
                "message": f"No details found for MID: {mid}"
            }), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parquet/persons', methods=['GET'])
def get_all_persons():
    """Get all person details with optional filtering"""
    try:
        # Get query parameters for filtering
        name_filter = request.args.get('name', '').strip()
        type_filter = request.args.get('type', '').strip()
        ranking_filter = request.args.get('ranking', '').strip()
        priority_filter = request.args.get('priority', '').strip()
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        with face_server.parquet_reader.file_lock:
            all_persons = list(face_server.parquet_reader.person_details.values())
            
            # Apply filters
            filtered_persons = all_persons
            
            if name_filter:
                filtered_persons = [p for p in filtered_persons 
                                 if name_filter.lower() in p['name'].lower()]
            
            if type_filter:
                filtered_persons = [p for p in filtered_persons 
                                 if type_filter.lower() == p['type'].lower()]
            
            if ranking_filter:
                filtered_persons = [p for p in filtered_persons 
                                  if ranking_filter.lower() in p['ranking'].lower()]
            
            if priority_filter:
                filtered_persons = [p for p in filtered_persons 
                                  if priority_filter.lower() == p['priority'].lower()]
            
            # Calculate pagination
            total_count = len(filtered_persons)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_persons = filtered_persons[start_idx:end_idx]
            
            return jsonify({
                "status": "success",
                "persons": paginated_persons,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": total_count,
                    "total_pages": (total_count + per_page - 1) // per_page,
                    "has_next": end_idx < total_count,
                    "has_prev": page > 1
                },
                "filters_applied": {
                    "name": name_filter,
                    "type": type_filter,
                    "ranking": ranking_filter,
                    "priority": priority_filter
                }
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parquet/persons', methods=['POST'])
def add_person_details():
    """Add person details - supports both single person and bulk operations"""
    try:
        # Validate request data
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Check if this is a bulk request (has 'persons' array) or single person request
        if 'persons' in request.json:
            # BULK OPERATION
            return handle_bulk_add_update(request.json['persons'])
        else:
            # SINGLE OPERATION
            return handle_single_add_update(request.json)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def handle_single_add_update(person_data):
    """Handle single person add/update"""
    required_fields = ['MID', 'Name', 'Ranking', 'Type', 'Priority']
    missing_fields = [field for field in required_fields if field not in person_data]
    
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
    
    # Extract and validate data
    mid = str(person_data['MID']).strip()
    name = str(person_data['Name']).strip()
    ranking = str(person_data['Ranking']).strip()
    type_val = str(person_data['Type']).strip()
    priority = str(person_data['Priority']).strip()
    
    if not all([mid, name, ranking, type_val, priority]):
        return jsonify({"error": "All fields must have non-empty values"}), 400
    
    # Validate Type and Priority values
    valid_types = ['Staff', 'Guest']
    valid_priorities = ['Low', 'Medium', 'High', 'Banned']
    
    if type_val not in valid_types:
        return jsonify({"error": f"Invalid Type. Must be one of: {valid_types}"}), 400
    
    if priority not in valid_priorities:
        return jsonify({"error": f"Invalid Priority. Must be one of: {valid_priorities}"}), 400
    
    # Check if MID already exists
    existing_details = face_server.parquet_reader.get_person_details(mid)
    
    if existing_details:
        # UPDATE existing person details
        result = face_server.parquet_reader.update_person_details(mid, name, ranking, type_val, priority)
        
        if result['success']:
            return jsonify({
                "status": "updated",
                "message": f"Person details updated for existing MID: {mid}",
                "action": "update",
                "previous_details": existing_details,
                "updated_details": {
                    "mid": mid,
                    "name": name,
                    "ranking": ranking,
                    "type": type_val,
                    "priority": priority
                },
                "total_records": result['total_records']
            })
        else:
            return jsonify({"error": f"Failed to update person details: {result['error']}"}), 500
    
    else:
        # ADD new person details
        result = face_server.parquet_reader.add_person_details(mid, name, ranking, type_val, priority)
        
        if result['success']:
            return jsonify({
                "status": "created",
                "message": f"New person details added for MID: {mid}",
                "action": "create",
                "person_details": {
                    "mid": mid,
                    "name": name,
                    "ranking": ranking,
                    "type": type_val,
                    "priority": priority
                },
                "total_records": result['total_records']
            })
        else:
            return jsonify({"error": f"Failed to add person details: {result['error']}"}), 500


def handle_bulk_add_update(persons_data):
    """Handle bulk person add/update operations"""
    if not isinstance(persons_data, list):
        return jsonify({"error": "Persons data must be a list"}), 400
    
    if len(persons_data) == 0:
        return jsonify({"error": "No persons provided in bulk request"}), 400
    
    if len(persons_data) > 1000:  # Limit bulk size
        return jsonify({"error": "Bulk request too large. Maximum 1000 persons per request"}), 400
    
    required_fields = ['MID', 'Name', 'Ranking', 'Type', 'Priority']
    valid_types = ['Staff', 'Guest']
    valid_priorities = ['Low', 'Medium', 'High', 'Banned']
    
    results = []
    errors = []
    created_count = 0
    updated_count = 0
    
    for idx, person in enumerate(persons_data):
        try:
            # Validate fields
            missing_fields = [field for field in required_fields if field not in person]
            if missing_fields:
                errors.append(f"Row {idx + 1}: Missing fields {missing_fields}")
                continue
            
            mid = str(person['MID']).strip()
            name = str(person['Name']).strip()
            ranking = str(person['Ranking']).strip()
            type_val = str(person['Type']).strip()
            priority = str(person['Priority']).strip()
            
            if not all([mid, name, ranking, type_val, priority]):
                errors.append(f"Row {idx + 1}: All fields must have non-empty values")
                continue
            
            # Validate Type and Priority
            if type_val not in valid_types:
                errors.append(f"Row {idx + 1}: Invalid Type '{type_val}'. Must be one of: {valid_types}")
                continue
            
            if priority not in valid_priorities:
                errors.append(f"Row {idx + 1}: Invalid Priority '{priority}'. Must be one of: {valid_priorities}")
                continue
            
            # Check if exists for update vs create
            existing_details = face_server.parquet_reader.get_person_details(mid)
            
            if existing_details:
                # Update existing
                result = face_server.parquet_reader.update_person_details(mid, name, ranking, type_val, priority)
                if result['success']:
                    updated_count += 1
                    results.append({
                        "mid": mid,
                        "action": "updated",
                        "previous_details": existing_details,
                        "updated_details": {
                            "mid": mid,
                            "name": name,
                            "ranking": ranking,
                            "type": type_val,
                            "priority": priority
                        }
                    })
                else:
                    errors.append(f"Row {idx + 1}: Failed to update MID '{mid}': {result['error']}")
            else:
                # Add new
                result = face_server.parquet_reader.add_person_details(mid, name, ranking, type_val, priority)
                if result['success']:
                    created_count += 1
                    results.append({
                        "mid": mid,
                        "action": "created",
                        "person_details": {
                            "mid": mid,
                            "name": name,
                            "ranking": ranking,
                            "type": type_val,
                            "priority": priority
                        }
                    })
                else:
                    errors.append(f"Row {idx + 1}: Failed to add MID '{mid}': {result['error']}")
                    
        except Exception as e:
            errors.append(f"Row {idx + 1}: {str(e)}")
    
    # Get final record count
    final_stats = face_server.parquet_reader.get_stats()
    
    return jsonify({
        "status": "success" if (created_count > 0 or updated_count > 0) else "failed",
        "message": f"Bulk operation completed: {created_count} created, {updated_count} updated",
        "bulk_operation": True,
        "total_processed": len(persons_data),
        "successful_operations": created_count + updated_count,
        "created_count": created_count,
        "updated_count": updated_count,
        "error_count": len(errors),
        "total_records": final_stats['total_records'],
        "results": results,
        "errors": errors if errors else None
    })

@app.route('/api/parquet/persons/<mid>', methods=['PUT'])
def update_person_details(mid):
    """Update existing person details"""
    try:
        # Check if person exists
        existing_details = face_server.parquet_reader.get_person_details(mid)
        if not existing_details:
            return jsonify({
                "error": f"Person with MID '{mid}' not found"
            }), 404
        
        # Validate request data
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get current details and update with provided fields
        updated_details = existing_details.copy()
        
        # Update only provided fields
        if 'Name' in request.json:
            updated_details['name'] = str(request.json['Name']).strip()
        if 'Ranking' in request.json:
            updated_details['ranking'] = str(request.json['Ranking']).strip()
        if 'Type' in request.json:
            updated_details['type'] = str(request.json['Type']).strip()
        if 'Priority' in request.json:
            updated_details['priority'] = str(request.json['Priority']).strip()
        
        # Validate that no field is empty
        if not all([updated_details['name'], updated_details['ranking'], 
                   updated_details['type'], updated_details['priority']]):
            return jsonify({"error": "All fields must have non-empty values"}), 400
        
        # Update in parquet file
        result = face_server.parquet_reader.update_person_details(
            mid, updated_details['name'], updated_details['ranking'],
            updated_details['type'], updated_details['priority']
        )
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": f"Person details updated for MID: {mid}",
                "previous_details": existing_details,
                "updated_details": updated_details,
                "total_records": result['total_records']
            })
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parquet/persons/<mid>', methods=['DELETE'])
def delete_person_details(mid):
    """Delete person details"""
    try:
        # Check if person exists
        existing_details = face_server.parquet_reader.get_person_details(mid)
        if not existing_details:
            return jsonify({
                "error": f"Person with MID '{mid}' not found"
            }), 404
        
        # Delete from parquet file
        result = face_server.parquet_reader.delete_person_details(mid)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": f"Person details deleted for MID: {mid}",
                "deleted_details": existing_details,
                "total_records": result['total_records']
            })
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parquet/persons/bulk', methods=['POST'])
def bulk_add_persons():
    """Bulk add multiple person details"""
    try:
        if not request.json or 'persons' not in request.json:
            return jsonify({"error": "No persons data provided"}), 400
        
        persons_data = request.json['persons']
        if not isinstance(persons_data, list):
            return jsonify({"error": "Persons data must be a list"}), 400
        
        required_fields = ['MID', 'Name', 'Ranking', 'Type', 'Priority']
        results = []
        errors = []
        
        for idx, person in enumerate(persons_data):
            try:
                # Validate fields
                missing_fields = [field for field in required_fields if field not in person]
                if missing_fields:
                    errors.append(f"Row {idx + 1}: Missing fields {missing_fields}")
                    continue
                
                mid = str(person['MID']).strip()
                name = str(person['Name']).strip()
                ranking = str(person['Ranking']).strip()
                type_val = str(person['Type']).strip()
                priority = str(person['Priority']).strip()
                
                if not all([mid, name, ranking, type_val, priority]):
                    errors.append(f"Row {idx + 1}: All fields must have non-empty values")
                    continue
                
                # Check if exists
                if face_server.parquet_reader.get_person_details(mid):
                    errors.append(f"Row {idx + 1}: MID '{mid}' already exists")
                    continue
                
                results.append({
                    "mid": mid,
                    "name": name,
                    "ranking": ranking,
                    "type": type_val,
                    "priority": priority
                })
                
            except Exception as e:
                errors.append(f"Row {idx + 1}: {str(e)}")
        
        if not results:
            return jsonify({
                "error": "No valid records to add",
                "errors": errors
            }), 400
        
        # Bulk add to parquet
        bulk_result = face_server.parquet_reader.bulk_add_persons(results)
        
        if bulk_result['success']:
            return jsonify({
                "status": "success",
                "message": f"Bulk added {len(results)} person records",
                "added_count": len(results),
                "total_records": bulk_result['total_records'],
                "errors": errors if errors else None
            })
        else:
            return jsonify({"error": bulk_result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parquet/persons/bulk', methods=['DELETE'])
def bulk_delete_persons():
    """Bulk delete multiple person details"""
    try:
        if not request.json or 'mids' not in request.json:
            return jsonify({"error": "No MIDs provided"}), 400
        
        mids = request.json['mids']
        if not isinstance(mids, list):
            return jsonify({"error": "MIDs must be a list"}), 400
        
        deleted_details = []
        not_found = []
        
        for mid in mids:
            existing = face_server.parquet_reader.get_person_details(str(mid).strip())
            if existing:
                deleted_details.append(existing)
            else:
                not_found.append(str(mid).strip())
        
        if not deleted_details:
            return jsonify({
                "error": "No valid MIDs found to delete",
                "not_found": not_found
            }), 404
        
        # Bulk delete
        bulk_result = face_server.parquet_reader.bulk_delete_persons([d['mid'] for d in deleted_details])
        
        if bulk_result['success']:
            return jsonify({
                "status": "success",
                "message": f"Bulk deleted {len(deleted_details)} person records",
                "deleted_count": len(deleted_details),
                "deleted_details": deleted_details,
                "not_found": not_found if not_found else None,
                "total_records": bulk_result['total_records']
            })
        else:
            return jsonify({"error": bulk_result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parquet/export', methods=['GET'])
def export_parquet_data():
    """Export parquet data as JSON or CSV"""
    try:
        export_format = request.args.get('format', 'json').lower()
        
        with face_server.parquet_reader.file_lock:
            all_persons = list(face_server.parquet_reader.person_details.values())
        
        if export_format == 'csv':
            import io
            output = io.StringIO()
            
            # Write CSV header
            output.write("MID,Name,Ranking,Type,Priority\n")
            
            # Write data
            for person in all_persons:
                output.write(f"{person['mid']},{person['name']},{person['ranking']},{person['type']},{person['priority']}\n")
            
            csv_data = output.getvalue()
            output.close()
            
            return csv_data, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename=person_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        
        else:  # JSON format
            return jsonify({
                "status": "success",
                "export_format": "json",
                "total_records": len(all_persons),
                "exported_at": datetime.now().isoformat(),
                "persons": all_persons
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
#==========================================================================================================================================================
@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Performance metrics with smart cache details"""
    try:
        return jsonify(face_server.get_performance_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize', methods=['POST', 'OPTIONS'])
def recognize():
    """Main recognition endpoint - all requests are high priority"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        request_id = str(uuid.uuid4())
        camera_id = request.json.get('camera_id')
        
        # Add to single high-priority queue
        try:
            face_server.recognition_queue.put_nowait((request_id, request.json['image'], camera_id))
        except queue.Full:
            return jsonify({
                "error": "Server at maximum capacity",
                "queue_size": face_server.recognition_queue.qsize()
            }), 503
        
        # Get result
        result = face_server.get_recognition_result(request_id, timeout=30)
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
        
        try:
            face_server.batch_request_queue.put_nowait((request_id, files_to_process))
        except queue.Full:
            return jsonify({"error": "Batch processing queue full"}), 503
        
        # Get results
        results = face_server.get_batch_result(request_id)
        
        return jsonify({
            "status": "success",
            "results": results,
            "total_files": len(files_to_process),
            "gpu_accelerated": True,
            "smart_caching_enabled": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_image_to_database', methods=['POST'])
def add_image_to_database():
    """Add image to database with cache clearing"""
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
        
        # Save and update database asynchronously
        def save_and_update():
            try:
                cv2.imwrite(original_filepath, img)
                
                # Add to database
                face_server.face_db[identity] = {
                    'embedding': embedding,
                    'image_path': original_filepath,
                    'timestamp': timestamp
                }
                
                # Save and rebuild (this will clear smart cache)
                face_server.save_embeddings()
                
            except Exception as e:
                print(f"[ERROR] Save and update failed: {e}")
        
        face_server.recognition_pool.submit(save_and_update)

        return jsonify({
            "status": "success",
            "message": f"Image for '{identity}' added to database",
            "filename": original_filename,
            "timestamp": timestamp,
            "gpu_accelerated": True,
            "cache_cleared": True
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

@app.route('/add_image_to_database_2', methods=['POST'])
def add_image_to_database_2():
    """Add image with multi-face handling and cache clearing"""
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

        # Save and update database asynchronously
        def save_and_update():
            try:
                cv2.imwrite(original_filepath, img)
                
                # Add to database
                face_server.face_db[identity] = {
                    'embedding': embedding,
                    'image_path': original_filepath,
                    'timestamp': timestamp
                }
                
                # Save and rebuild (this will clear smart cache)
                face_server.save_embeddings()
                
            except Exception as e:
                print(f"[ERROR] Save and update failed: {e}")
        
        face_server.recognition_pool.submit(save_and_update)

        response_data = {
            "status": "success",
            "message": f"Image for '{identity}' added to database",
            "filename": original_filename,
            "timestamp": timestamp,
            "gpu_accelerated": True,
            "cache_cleared": True
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
    """Top matches recognition with smart caching"""
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
        
        # Check smart cache first
        cache_found, cache_result, cache_similarity, cache_type = face_server.smart_cache.search_in_cache(
            query_embedding, threshold=min_threshold
        )
        
        top_matches = []
        
        if cache_found:
            # Add cache result as top match
            top_matches.append({
                "identity": cache_result.get('identity', ''),
                "confidence": float(cache_similarity),
                "matched_image": cache_result.get('matched_image', ''),
                "source": "cache"
            })
        
        # Search database for additional matches
        search_results = face_server.safe_faiss_search(query_embedding, top_k=top_k * 2)
        
        # Fallback to numpy
        if not search_results:
            search_results = face_server.numpy_similarity_search(query_embedding, top_k=top_k * 2)
        
        # Add database results
        for identity, similarity in search_results:
            if similarity >= min_threshold and len(top_matches) < top_k:
                # Skip if already in cache results
                if cache_found and identity.split('@')[0] == cache_result.get('identity', ''):
                    continue
                    
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
                    "matched_image": matched_image_base64,
                    "source": "database"
                })

        return jsonify({
            "status": "success",
            "matches": top_matches,
            "cache_hit": cache_found,
            "gpu_accelerated": True,
            "smart_caching_enabled": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recognition_events', methods=['GET', 'OPTIONS'])
def get_recognition_events():
    """Get recognition events with cache information"""
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
            "gpu_accelerated": True,
            "smart_caching_enabled": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reload_embeddings', methods=['POST'])
def reload_embeddings():
    """Reload embeddings and clear smart cache"""
    try:
        face_server.face_db = face_server.load_embeddings()
        
        # Clear smart cache when reloading
        cleared_count = face_server.smart_cache.clear_cache()
        print(f"[CACHE] Cleared {cleared_count} cache entries due to reload")
        
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
            "message": "Embeddings reloaded, cache cleared, and index rebuilding",
            "embedding_count": len(face_server.face_db),
            "cache_cleared": cleared_count,
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
            return jsonify({
                "status": "success", 
                "message": f"Threshold set to {new_threshold}",
                "cache_threshold": face_server.cache_threshold
            })
        else:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid threshold value"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Smart cache statistics"""
    try:
        cache_stats = face_server.smart_cache.get_stats()
        
        return jsonify({
            "smart_cache": cache_stats,
            "cache_threshold": face_server.cache_threshold,
            "gpu_accelerated": True,
            "smart_caching_enabled": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_caches():
    """Clear smart cache"""
    try:
        cleared_count = face_server.smart_cache.clear_cache()
        
        return jsonify({
            "status": "success",
            "message": f"Cleared smart cache with {cleared_count} entries",
            "gpu_accelerated": True,
            "smart_caching_enabled": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/set_threshold', methods=['POST'])
def set_cache_threshold():
    """Set cache similarity threshold"""
    try:
        new_threshold = float(request.json['threshold'])
        if 0 <= new_threshold <= 1:
            face_server.cache_threshold = new_threshold
            return jsonify({
                "status": "success", 
                "message": f"Cache threshold set to {new_threshold}",
                "recognition_threshold": face_server.threshold,
                "cache_threshold": face_server.cache_threshold
            })
        else:
            return jsonify({"error": "Cache threshold must be between 0 and 1"}), 400
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid threshold value"}), 400
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
            "gpu_available": torch.cuda.is_available(),
            "smart_caching_enabled": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 28. FILE SERVING ROUTES
@app.route('/Get/In/<path:filename>')
def serve_get_in_files(filename):
    get_in_folder = os.path.join(os.getcwd(), 'Get', 'In')
    return send_from_directory(get_in_folder, filename)

@app.route('/Database/images/<path:filename>')
def serve_Database_files(filename):
    get_in_folder = os.path.join(os.getcwd(), 'Database', 'images')
    return send_from_directory(get_in_folder, filename)

# 29. STATIC ROUTES
@app.route('/realtime')
def realtime_monitor():
    return app.send_static_file('realtime.html')

@app.route('/Check1')
def Check1_monitor():
    return app.send_static_file('Check1.html')

@app.route('/Check2')
def Check2_monitor():
    return app.send_static_file('Check2.html')

@app.route('/cache_monitor')
def cache_monitor():
    return app.send_static_file('cache_monitor.html')

@app.route('/details')
def details_monitor():
    return app.send_static_file('details.html')

# 30. ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# 31. MAIN EXECUTION
if __name__ == '__main__':
    print("🚀 Starting Optimized Face Recognition Server with Smart Caching...")
    print(f"⚡ GPU Memory Reserved: {face_server.gpu_memory_limit / (1024**3):.1f}GB")
    print(f"💾 {len(face_server.face_db)} face embeddings loaded")
    print(f"🔍 FAISS GPU Index: {face_server.faiss_index.ntotal if face_server.faiss_index else 0} entries")
    print(f"🧠 Smart Cache: {face_server.smart_cache.max_size} entries, {face_server.smart_cache.duration} duration")
    print(f"🎯 Cache Threshold: {face_server.cache_threshold * 100}% similarity")
    print(f"🎯 Recognition Threshold: {face_server.threshold * 100}% similarity")
    print(f"🚀 All requests are HIGH PRIORITY - No queue levels!")
    print(f"💨 Smart Caching Flow:")
    print(f"   1. Extract embedding from image")
    print(f"   2. Check smart cache (≥50% similarity)")
    print(f"   3. If cache HIT → Return cached person + new confidence")
    print(f"   4. If cache MISS → Search database")
    print(f"   5. If DB match (≥50%) → Store in cache + return")
    print(f"   6. If no match → Handle as unrecognized")
    print(f"🔥 Maximum Performance with Smart Caching Ready!")
    
    # Run the server
    app.run(host='0.0.0.0', port=3005, debug=False, threaded=True)
