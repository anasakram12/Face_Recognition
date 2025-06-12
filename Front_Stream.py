import cv2
import threading
import time
import queue
import logging
import numpy as np
from flask import Flask, Response, render_template_string, request
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rtsp_streaming')

# Camera settings
camera_sources = {
    'cam1': {
        'url': 'rtsp://admin:Think22wise@192.168.15.31:554/Streaming/Channels/102',
        'name': 'Travel Counter Out'
    },
    'cam2': {
        'url': 'rtsp://admin:Think22wise@192.168.15.32:554/Streaming/Channels/102',
        'name': 'Travel Counter In'
    },
    'cam3': {
        'url': 'rtsp://admin:Think22wise@192.168.15.61:554/Streaming/Channels/102',
        'name': 'Membership Counter Out'
    },
    'cam4': {
        'url': 'rtsp://admin:Think22wise@192.168.15.62:554/Streaming/Channels/102',
        'name': 'Membership Counter In'
    }
}

app = Flask(__name__)

# Global frame storage
frame_queues = {
    'cam1': queue.Queue(maxsize=10),
    'cam2': queue.Queue(maxsize=10),
    'cam3': queue.Queue(maxsize=10),
    'cam4': queue.Queue(maxsize=10)
}

# Camera status
camera_status = {
    'cam1': {'connected': False, 'last_frame_time': 0},
    'cam2': {'connected': False, 'last_frame_time': 0},
    'cam3': {'connected': False, 'last_frame_time': 0},
    'cam4': {'connected': False, 'last_frame_time': 0}
}

# Thread to grab frames from cameras
def camera_stream_thread(camera_id):
    url = camera_sources[camera_id]['url']
    name = camera_sources[camera_id]['name']
    
    while True:
        try:
            logger.info(f"Connecting to camera {camera_id}: {name} at {url}")
            camera_status[camera_id]['connected'] = False
            cap = cv2.VideoCapture(url)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}: {name}")
                # No longer adding error frames to the queue
                time.sleep(2)  # Wait before reconnection
                continue
            
            logger.info(f"Successfully connected to camera {camera_id}: {name}")
            camera_status[camera_id]['connected'] = True
            
            # Read frames in a loop
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to get frame from camera {camera_id}: {name}")
                    break
                
                # Process frame - resize for consistency if needed
                frame = cv2.resize(frame, (640, 480))
                
                # Add timestamp and camera name
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"{name} - {timestamp}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Update the frame queue
                if frame_queues[camera_id].full():
                    try:
                        frame_queues[camera_id].get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    frame_queues[camera_id].put_nowait(frame)
                    camera_status[camera_id]['last_frame_time'] = time.time()
                except queue.Full:
                    pass
                
                # Small sleep to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error with camera {camera_id}: {str(e)}")
            
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            
            camera_status[camera_id]['connected'] = False
            logger.info(f"Camera {camera_id} disconnected. Attempting to reconnect in 2 seconds...")
            
            # No longer adding error frames to the queue
            # Just clear the queue to avoid displaying stale frames
            while not frame_queues[camera_id].empty():
                try:
                    frame_queues[camera_id].get_nowait()
                except queue.Empty:
                    break
            
            time.sleep(2)  # Wait before reconnection

def generate_frames(camera_id):
    """Generator function to yield frames from the specified camera's queue."""
    while True:
        if not frame_queues[camera_id].empty() and camera_status[camera_id]['connected']:
            frame = frame_queues[camera_id].get()
            ret, buffer = cv2.imencode('.jpg', frame)
            
            if not ret:
                time.sleep(0.033)  # ~30 FPS
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If queue is empty or camera is disconnected, send an empty response
            # This will trigger the error handler in the frontend to show the loader
            time.sleep(0.033)  # ~30 FPS
            continue

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Route to access the camera stream."""
    if camera_id in camera_sources:
        return Response(generate_frames(camera_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid camera ID", 404

@app.route('/api/camera_status')
def get_camera_status():
    """API endpoint to get the status of all cameras."""
    status_data = {}
    
    for cam_id, status in camera_status.items():
        # Calculate how long ago the last frame was received
        if status['last_frame_time'] > 0:
            time_since_last_frame = time.time() - status['last_frame_time']
            status_data[cam_id] = {
                'connected': status['connected'] and time_since_last_frame < 5,  # Consider disconnected if no frame for 5 seconds
                'name': camera_sources[cam_id]['name'],
                'time_since_last_frame': f"{time_since_last_frame:.1f}s"
            }
        else:
            status_data[cam_id] = {
                'connected': False,
                'name': camera_sources[cam_id]['name'],
                'time_since_last_frame': "Never"
            }
    
    return status_data

@app.route('/')
def index():
    """Main route to display the camera streams."""
    return render_template_string(HTML_TEMPLATE)

# HTML Template with styling similar to the provided example
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Monitoring System</title>
    <style>
        /* Modern Gold and Red Theme similar to SmartFace Sentinel */
        /* Modern Gold and Red Theme similar to SmartFace Sentinel */        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(to bottom, #2e0202, #4a0303);
            min-height: 100vh;
            color: #f5d76e;
            display: flex;
            flex-direction: column;
        }

        .container {
            width: 100%;
            max-width: none;
            margin: 0;
            flex-grow: 1;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            animation: fadeIn 1s ease-in-out;
        }

        h1 {
            font-size: 2.5rem;
            color: #f5d76e;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 3px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }

        .subtitle {
            font-size: 1.2rem;
            color: #e6c568;
            opacity: 0.9;
            margin-bottom: 20px;
        }

        .camera-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0;
            margin-bottom: 30px;
        }

        .camera-grid.grid-2 {
            grid-template-columns: repeat(2, 1fr);
        }

        .camera-grid.grid-3 {
            grid-template-columns: repeat(3, 1fr);
        }

        .camera-grid.grid-4 {
            grid-template-columns: repeat(4, 1fr);
        }

        .camera-grid.grid-5 {
            grid-template-columns: repeat(5, 1fr);
        }

        .camera-card {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 0;
            overflow: hidden;
            box-shadow: none;
            transition: all 0.3s ease;
            animation: slideUp 0.8s ease-out;
            position: relative;
            margin: 0;
        }

        .camera-card:hover {
            transform: none;
            box-shadow: none;
            border: none;
        }

        .camera-header {
            background: #4a0303;
            padding: 15px;
            text-align: center;
            border-bottom: 2px solid #f5d76e;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .camera-title {
            font-size: 12px;
            color: #f5d76e;
            margin: 0;
            letter-spacing: 1px;
        }

        .camera-status {
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4caf50;
            animation: pulse 2s infinite;
        }

        .status-dot.offline {
            background: #f44336;
        }

        .camera-feed-container {
            position: relative;
            width: 100%;
            height: 0;
            padding-bottom: 50%;
            background-color: #000;
        }

        .camera-feed {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            cursor: pointer;
        }

        .loader-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 10;
        }

        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #f5d76e;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
            margin-bottom: 15px;
        }

        .loader-text {
            color: #f5d76e;
            font-size: 1.2rem;
            text-align: center;
        }

        .system-status {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px 15px;
            border-radius: 20px;
        }

        .status-text {
            font-size: 0.9rem;
            font-weight: bold;
        }

        .live-time {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
            color: #f5d76e;
        }

        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .control-button {
            padding: 10px 20px;
            background: rgba(0, 0, 0, 0.5);
            color: #f5d76e;
            border: 1px solid #f5d76e;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
        }

        .control-button:hover {
            background: #f5d76e;
            color: #4a0303;
        }

        .control-button.active {
            background: #f5d76e;
            color: #4a0303;
        }

        .footer {
            text-align: center;
            padding: 20px 0;
            font-size: 0.9rem;
            color: #e6c568;
            opacity: 0.7;
            margin-top: auto;
        }

        .fullscreen-indicator {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.6);
            color: #f5d76e;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.8rem;
            z-index: 5;
            display: none;
        }

        .camera-feed-container:hover .fullscreen-indicator {
            display: block;
        }

        /* Home button styling */
        .home-button {
            padding: 10px 20px;
            background: #f5d76e;
            color: #4a0303;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
            text-decoration: none;
            display: inline-block;
            margin-right: 15px;
        }

        .home-button:hover {
            background: #e6c568;
            transform: translateY(-2px);
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive layout */
        @media (max-width: 1200px) {
            .camera-grid.grid-5 {
                grid-template-columns: repeat(4, 1fr);
            }
        }

        @media (max-width: 992px) {
            .camera-grid.grid-3, .camera-grid.grid-4, .camera-grid.grid-5 {
                grid-template-columns: repeat(2, 1fr);
            }
            
            h1 {
                font-size: 2rem;
            }
        }

        @media (max-width: 768px) {
            .camera-grid {
                grid-template-columns: 1fr;
            }
            
            .camera-grid.grid-2, .camera-grid.grid-3, .camera-grid.grid-4, .camera-grid.grid-5 {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="system-status">
            <div class="status-dot" id="system-status-dot"></div>
            <span class="status-text" id="system-status-text">System Status</span>
        </div>

        <div class="live-time" id="live-clock">
            --:--:--
        </div>

        <div class="header">
            <h1> SmartFace Sentinel Live Camera Monitoring </h1>
            <div class="subtitle">Real-time Surveillance Dashboard</div>
        </div>

        <div class="controls">
            <a href="http://192.168.15.65:3006/" class="home-button">Home</a>
            <button class="control-button" id="refresh-button">Refresh Streams</button>
            <button class="control-button" id="fullscreen-button">Toggle Fullscreen</button>
            
            <!-- Grid layout control buttons -->
            <button class="control-button grid-button active" data-grid="2">2 Column</button>
            <button class="control-button grid-button" data-grid="3">3 Column</button>
            <button class="control-button grid-button" data-grid="4">4 Column</button>
            <button class="control-button grid-button" data-grid="5">5 Column</button>
        </div>

        <div class="camera-grid grid-2">
            <!-- Original 4 cameras -->
            <div class="camera-card" style="animation-delay: 0.1s;">
                <div class="camera-header">
                    <h3 class="camera-title">Travel Counter Out</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam1"></div>
                        <span class="status-text" id="status-text-cam1">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="/video_feed/cam1" class="camera-feed" id="feed-cam1" alt="Camera 1 Feed">
                    <div class="loader-container" id="loader-cam1">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <div class="camera-card" style="animation-delay: 0.2s;">
                <div class="camera-header">
                    <h3 class="camera-title">Travel Counter In</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam2"></div>
                        <span class="status-text" id="status-text-cam2">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="/video_feed/cam2" class="camera-feed" id="feed-cam2" alt="Camera 2 Feed">
                    <div class="loader-container" id="loader-cam2">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <div class="camera-card" style="animation-delay: 0.3s;">
                <div class="camera-header">
                    <h3 class="camera-title">Membership Counter Out</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam3"></div>
                        <span class="status-text" id="status-text-cam3">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="/video_feed/cam3" class="camera-feed" id="feed-cam3" alt="Camera 3 Feed">
                    <div class="loader-container" id="loader-cam3">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <div class="camera-card" style="animation-delay: 0.4s;">
                <div class="camera-header">
                    <h3 class="camera-title">Membership Counter In</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam4"></div>
                        <span class="status-text" id="status-text-cam4">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="/video_feed/cam4" class="camera-feed" id="feed-cam4" alt="Camera 4 Feed">
                    <div class="loader-container" id="loader-cam4">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <!-- Paiza Gate cameras -->
            <div class="camera-card" style="animation-delay: 0.5s;">
                <div class="camera-header">
                    <h3 class="camera-title">Paiza Gate 1 (Stream 1)</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam5"></div>
                        <span class="status-text" id="status-text-cam5">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="http://192.168.15.131:5050/video_feed" class="camera-feed" id="feed-cam5" alt="Paiza Gate 1 Stream 1">
                    <div class="loader-container" id="loader-cam5">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <div class="camera-card" style="animation-delay: 0.6s;">
                <div class="camera-header">
                    <h3 class="camera-title">Paiza Gate 1 (Stream 2)</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam6"></div>
                        <span class="status-text" id="status-text-cam6">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="http://192.168.15.131:5051/video_feed" class="camera-feed" id="feed-cam6" alt="Paiza Gate 1 Stream 2">
                    <div class="loader-container" id="loader-cam6">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <div class="camera-card" style="animation-delay: 0.7s;">
                <div class="camera-header">
                    <h3 class="camera-title">Paiza Gate 2 (Stream 1)</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam7"></div>
                        <span class="status-text" id="status-text-cam7">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="http://192.168.15.130:5050/video_feed" class="camera-feed" id="feed-cam7" alt="Paiza Gate 2 Stream 1">
                    <div class="loader-container" id="loader-cam7">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <div class="camera-card" style="animation-delay: 0.8s;">
                <div class="camera-header">
                    <h3 class="camera-title">Paiza Gate 2 (Stream 2)</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam8"></div>
                        <span class="status-text" id="status-text-cam8">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="http://192.168.15.130:5051/video_feed" class="camera-feed" id="feed-cam8" alt="Paiza Gate 2 Stream 2">
                    <div class="loader-container" id="loader-cam8">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <!-- New Table 37 Cameras -->
            <div class="camera-card" style="animation-delay: 0.9s;">
                <div class="camera-header">
                    <h3 class="camera-title">Table 37 (Left)</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam9"></div>
                        <span class="status-text" id="status-text-cam9">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="http://192.168.15.28:5050/video_feed" class="camera-feed" id="feed-cam9" alt="Table 37 Left View">
                    <div class="loader-container" id="loader-cam9">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
            
            <div class="camera-card" style="animation-delay: 1.0s;">
                <div class="camera-header">
                    <h3 class="camera-title">Table 37 (Right)</h3>
                    <div class="camera-status">
                        <div class="status-dot offline" id="status-cam10"></div>
                        <span class="status-text" id="status-text-cam10">Connecting...</span>
                    </div>
                </div>
                <div class="camera-feed-container">
                    <img src="http://192.168.15.28:5051/video_feed" class="camera-feed" id="feed-cam10" alt="Table 37 Right View">
                    <div class="loader-container" id="loader-cam10">
                        <div class="loader"></div>
                        <div class="loader-text">Connecting to stream...</div>
                    </div>
                    <div class="fullscreen-indicator">Click to toggle fullscreen</div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        © 2025 Sentinel Live Feeds | Last Updated: <span id="last-update"></span>
    </div>

    <script>
        // Initialize the dashboard
        document.addEventListener('DOMContentLoaded', function() {
            // Set the last update time
            updateDateTime();
            
            // Start the live clock
            setInterval(updateLiveClock, 1000);
            
            // Check camera status periodically
            checkCameraStatus();
            setInterval(checkCameraStatus, 5000);
            
            // Button event listeners
            document.getElementById('refresh-button').addEventListener('click', refreshStreams);
            document.getElementById('fullscreen-button').addEventListener('click', toggleFullscreen);

            // Setup grid layout buttons
            setupGridButtons();

            // Setup fullscreen toggle for camera feeds
            setupFullscreenToggles();

            // Setup error handlers for streams
            setupStreamErrorHandlers();
        });

        // Update the live clock
        function updateLiveClock() {
            const now = new Date();
            const hours = String(now.getHours()).padStart(2, '0');
            const minutes = String(now.getMinutes()).padStart(2, '0');
            const seconds = String(now.getSeconds()).padStart(2, '0');
            document.getElementById('live-clock').textContent = `${hours}:${minutes}:${seconds}`;
        }

        // Update the last updated date and time
        function updateDateTime() {
            const now = new Date();
            document.getElementById('last-update').textContent = now.toLocaleString();
        }

        // Setup grid layout buttons
        function setupGridButtons() {
            document.querySelectorAll('.grid-button').forEach(button => {
                button.addEventListener('click', function() {
                    // Remove active class from all grid buttons
                    document.querySelectorAll('.grid-button').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    
                    // Add active class to clicked button
                    this.classList.add('active');
                    
                    // Get the grid layout
                    const gridLayout = this.getAttribute('data-grid');
                    
                    // Update the camera grid class
                    const cameraGrid = document.querySelector('.camera-grid');
                    cameraGrid.className = `camera-grid grid-${gridLayout}`;
                });
            });
        }

        // Check the status of all cameras
        async function checkCameraStatus() {
            try {
                const response = await fetch('/api/camera_status');
                const statusData = await response.json();
                
                let allConnected = true;
                
                // Update statuses for the first 4 cameras from the API
                for (const camId in statusData) {
                    const statusDot = document.getElementById(`status-${camId}`);
                    const statusText = document.getElementById(`status-text-${camId}`);
                    const loaderContainer = document.getElementById(`loader-${camId}`);
                    
                    if (statusData[camId].connected) {
                        statusDot.classList.remove('offline');
                        statusText.textContent = 'Online';
                        loaderContainer.style.display = 'none';
                    } else {
                        statusDot.classList.add('offline');
                        statusText.textContent = 'Connecting...';
                        loaderContainer.style.display = 'flex';
                        allConnected = false;
                    }
                }
                
                // For the additional streams (5-10), we need to manually check
                checkExternalCameraStatus();
                
            } catch (error) {
                console.error('Error checking camera status:', error);
                // If we can't reach the server, show all cameras as offline
                for (let i = 1; i <= 10; i++) {
                    const camId = `cam${i}`;
                    const statusDot = document.getElementById(`status-${camId}`);
                    const statusText = document.getElementById(`status-text-${camId}`);
                    const loaderContainer = document.getElementById(`loader-${camId}`);
                    
                    statusDot.classList.add('offline');
                    statusText.textContent = 'Server Error';
                    loaderContainer.style.display = 'flex';
                    
                    const loaderText = loaderContainer.querySelector('.loader-text');
                    loaderText.textContent = 'Connection to server lost...';
                }
                
                updateSystemStatus();
            }
        }

        // Check external camera status (for Paiza Gate cameras and Table 37 cameras)
        function checkExternalCameraStatus() {
            // For cameras 5-10 (external cameras), we'll check if the image has loaded
            for (let i = 5; i <= 10; i++) {
                const camId = `cam${i}`;
                const feed = document.getElementById(`feed-${camId}`);
                const statusDot = document.getElementById(`status-${camId}`);
                const statusText = document.getElementById(`status-text-${camId}`);
                
                // If the image has a natural width, it's loaded
                if (feed.naturalWidth > 0) {
                    statusDot.classList.remove('offline');
                    statusText.textContent = 'Online';
                    document.getElementById(`loader-${camId}`).style.display = 'none';
                } else {
                    statusDot.classList.add('offline');
                    statusText.textContent = 'Connecting...';
                    document.getElementById(`loader-${camId}`).style.display = 'flex';
                }
            }
            
            updateSystemStatus();
        }

        // Update the overall system status
        function updateSystemStatus() {
            const systemStatusDot = document.getElementById('system-status-dot');
            const systemStatusText = document.getElementById('system-status-text');
            
            // Check if any cameras are offline
            const hasOffline = Array.from(document.querySelectorAll('.status-dot'))
                                   .some(dot => dot.classList.contains('offline') && 
                                               !dot.id.includes('system-status'));
            
            if (!hasOffline) {
                systemStatusDot.classList.remove('offline');
                systemStatusText.textContent = 'All Cameras Online';
            } else {
                systemStatusDot.classList.add('offline');
                systemStatusText.textContent = 'Some Cameras Offline';
            }
        }

        // Refresh all camera streams
        function refreshStreams() {
            const feeds = document.querySelectorAll('.camera-feed');
            feeds.forEach(feed => {
                const src = feed.src;
                feed.src = '';
                
                // Show loader while refreshing
                const camId = feed.id.split('-')[1];
                document.getElementById(`loader-${camId}`).style.display = 'flex';
                
                setTimeout(() => {
                    // Add cache-busting parameter for local feeds
                    if (src.includes('/video_feed/')) {
                        feed.src = src.split('?')[0] + '?' + new Date().getTime();
                    } 
                    // For external feeds we just reload
                    else {
                        feed.src = src;
                    }
                }, 100);
            });
            
            updateDateTime();
        }

        // Setup fullscreen toggle for each camera
        function setupFullscreenToggles() {
            document.querySelectorAll('.camera-feed').forEach(feed => {
                feed.addEventListener('click', function() {
                    const container = this.closest('.camera-feed-container');
                    
                    if (!document.fullscreenElement) {
                        // Enter fullscreen
                        if (container.requestFullscreen) {
                            container.requestFullscreen();
                        } else if (container.webkitRequestFullscreen) {
                            container.webkitRequestFullscreen();
                        } else if (container.msRequestFullscreen) {
                            container.msRequestFullscreen();
                        }
                    } else {
                        // Exit fullscreen
                        if (document.exitFullscreen) {
                            document.exitFullscreen();
                        } else if (document.webkitExitFullscreen) {
                            document.webkitExitFullscreen();
                        } else if (document.msExitFullscreen) {
                            document.msExitFullscreen();
                        }
                    }
                });
            });

            // Listen for fullscreen change events
            document.addEventListener('fullscreenchange', handleFullscreenChange);
            document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
            document.addEventListener('mozfullscreenchange', handleFullscreenChange);
            document.addEventListener('MSFullscreenChange', handleFullscreenChange);
        }

        // Handle fullscreen change events
        function handleFullscreenChange() {
            const fullscreenElement = document.fullscreenElement || 
                                     document.webkitFullscreenElement || 
                                     document.mozFullscreenElement || 
                                     document.msFullscreenElement;
                                     
            // Update fullscreen indicators based on state
            document.querySelectorAll('.fullscreen-indicator').forEach(indicator => {
                if (fullscreenElement) {
                    indicator.textContent = 'Click to exit fullscreen';
                } else {
                    indicator.textContent = 'Click to toggle fullscreen';
                }
            });
        }

        // Setup error handlers for video streams
        function setupStreamErrorHandlers() {
            document.querySelectorAll('.camera-feed').forEach(feed => {
                feed.addEventListener('error', function() {
                    const camId = this.id.split('-')[1];
                    const loaderContainer = document.getElementById(`loader-${camId}`);
                    const loaderText = loaderContainer.querySelector('.loader-text');
                    const statusDot = document.getElementById(`status-${camId}`);
                    const statusText = document.getElementById(`status-text-${camId}`);
                    
                    loaderContainer.style.display = 'flex';
                    loaderText.textContent = 'Stream error. Reconnecting...';
                    
                    statusDot.classList.add('offline');
                    statusText.textContent = 'Connection Error';
                    
                    // Try to reload the stream after a delay
                    setTimeout(() => {
                        this.src = this.src.split('?')[0] + '?' + new Date().getTime();
                    }, 3000);
                    
                    updateSystemStatus();
                });
                
                // Hide loader when image loads successfully
                feed.addEventListener('load', function() {
                    const camId = this.id.split('-')[1];
                    document.getElementById(`loader-${camId}`).style.display = 'none';
                    
                    // Update the status dot and text
                    const statusDot = document.getElementById(`status-${camId}`);
                    const statusText = document.getElementById(`status-text-${camId}`);
                    
                    statusDot.classList.remove('offline');
                    statusText.textContent = 'Online';
                    
                    updateSystemStatus();
                });
            });
        }

        // Toggle fullscreen for the entire dashboard
        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(err => {
                    console.log(`Error attempting to enable fullscreen: ${err.message}`);
                });
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
            }
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Start camera threads before running the app
    for cam_id in camera_sources:
        thread = threading.Thread(target=camera_stream_thread, args=(cam_id,), daemon=True)
        thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=3007, debug=False, threaded=True)