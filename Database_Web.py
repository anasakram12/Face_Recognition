import os
import cv2
import pickle
import shutil
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
DATABASE_PATH = './Database/face_embeddings.pkl'
BACKUP_FOLDER = './Database/backups'
IMAGES_FOLDER = './Database/images'
MAX_BACKUPS = 5

# Ensure directories exist
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Global variables
face_db = {}
db_lock = threading.Lock()

def load_database():
    """Load the face database from pickle file"""
    global face_db
    try:
        if os.path.exists(DATABASE_PATH):
            with open(DATABASE_PATH, 'rb') as f:
                data = pickle.load(f)
                face_db = data.get('embeddings', {})
        else:
            face_db = {}
        print(f"Database loaded with {len(face_db)} entries")
    except Exception as e:
        print(f"Error loading database: {e}")
        face_db = {}

def save_database():
    """Save the face database to pickle file with backup"""
    global face_db
    try:
        with db_lock:
            # Create backup before saving
            create_backup()
            
            # Save the updated database
            data = {
                'embeddings': face_db,
                'last_modified': datetime.now()
            }
            
            temp_path = DATABASE_PATH + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Atomic replace
            if os.path.exists(DATABASE_PATH):
                os.replace(temp_path, DATABASE_PATH)
            else:
                os.rename(temp_path, DATABASE_PATH)
                
        print("Database saved successfully")
        return True
    except Exception as e:
        print(f"Error saving database: {e}")
        return False

def create_backup():
    """Create a timestamped backup of the current database"""
    try:
        if not os.path.exists(DATABASE_PATH):
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"face_embeddings_{timestamp}.pkl"
        backup_path = os.path.join(BACKUP_FOLDER, backup_filename)
        
        shutil.copy2(DATABASE_PATH, backup_path)
        print(f"Backup created: {backup_filename}")
        
        # Clean old backups (keep only last MAX_BACKUPS files)
        cleanup_old_backups()
        
    except Exception as e:
        print(f"Error creating backup: {e}")

def cleanup_old_backups():
    """Remove old backup files, keeping only the most recent MAX_BACKUPS"""
    try:
        backup_files = []
        for filename in os.listdir(BACKUP_FOLDER):
            if filename.startswith('face_embeddings_') and filename.endswith('.pkl'):
                filepath = os.path.join(BACKUP_FOLDER, filename)
                backup_files.append((filepath, os.path.getctime(filepath)))
        
        # Sort by creation time (newest first)
        backup_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old files
        for filepath, _ in backup_files[MAX_BACKUPS:]:
            os.remove(filepath)
            print(f"Removed old backup: {os.path.basename(filepath)}")
            
    except Exception as e:
        print(f"Error cleaning up backups: {e}")

def get_image_base64(image_path):
    """Convert image to base64 for web display"""
    try:
        if not os.path.exists(image_path):
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Resize image for web display
        img = cv2.resize(img, (200, 200))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('database.html')

@app.route('/api/entries')
def get_entries():
    """Get paginated entries with search functionality"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 100))
        search = request.args.get('search', '').lower()
        
        # Filter entries based on search
        filtered_entries = []
        for id_name, data in face_db.items():
            if search in id_name.lower():
                filtered_entries.append({
                    'id': id_name,
                    'image_path': data.get('image_path', ''),
                    'filename': os.path.basename(data.get('image_path', ''))
                })
        
        # Sort entries
        filtered_entries.sort(key=lambda x: x['id'])
        
        # Pagination
        total = len(filtered_entries)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        entries = filtered_entries[start_idx:end_idx]
        
        # Add image data for current page
        for entry in entries:
            entry['image_data'] = get_image_base64(entry['image_path'])
        
        return jsonify({
            'entries': entries,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_id', methods=['POST'])
def update_id():
    """Update an entry's ID and automatically rename the file"""
    try:
        data = request.json
        old_id = data.get('old_id')
        new_id = data.get('new_id', '').strip()
        
        if not old_id or not new_id:
            return jsonify({'error': 'Invalid ID provided'}), 400
            
        if old_id == new_id:
            return jsonify({'success': True})
            
        if new_id in face_db:
            return jsonify({'error': 'ID already exists'}), 400
            
        if old_id not in face_db:
            return jsonify({'error': 'Original ID not found'}), 404
        
        # Get the current file path and extract extension
        old_path = face_db[old_id]['image_path']
        file_extension = os.path.splitext(old_path)[1]  # Get .jpg, .png, etc.
        
        # Create new filename with the new ID
        new_filename = f"{new_id}{file_extension}"
        new_path = os.path.join(os.path.dirname(old_path), new_filename)
        
        # Check if the new filename already exists
        if os.path.exists(new_path):
            return jsonify({'error': f'File "{new_filename}" already exists'}), 400
        
        # Rename the physical file
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
        
        # Update the database entry
        face_db[new_id] = face_db.pop(old_id)
        face_db[new_id]['image_path'] = new_path
        
        if save_database():
            return jsonify({
                'success': True, 
                'new_filename': new_filename,
                'message': f'ID updated from "{old_id}" to "{new_id}" and file renamed to "{new_filename}"'
            })
        else:
            # Rollback file rename if database save fails
            if os.path.exists(new_path):
                shutil.move(new_path, old_path)
            return jsonify({'error': 'Failed to save database'}), 500
            
    except Exception as e:
        # Rollback file rename if any error occurs
        try:
            if 'new_path' in locals() and 'old_path' in locals():
                if os.path.exists(new_path) and not os.path.exists(old_path):
                    shutil.move(new_path, old_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500

@app.route('/api/rename_file', methods=['POST'])
def rename_file():
    """Rename an image file"""
    try:
        data = request.json
        id_name = data.get('id')
        new_filename = secure_filename(data.get('new_filename', ''))
        
        if not id_name or not new_filename:
            return jsonify({'error': 'Invalid parameters'}), 400
            
        if id_name not in face_db:
            return jsonify({'error': 'ID not found'}), 404
            
        old_path = face_db[id_name]['image_path']
        new_path = os.path.join(os.path.dirname(old_path), new_filename)
        
        if os.path.exists(new_path):
            return jsonify({'error': 'File name already exists'}), 400
            
        # Rename the file
        shutil.move(old_path, new_path)
        
        # Update database
        face_db[id_name]['image_path'] = new_path
        
        if save_database():
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Failed to save database'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/replace_image', methods=['POST'])
def replace_image():
    """Replace an image file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        id_name = request.form.get('id')
        
        if not id_name or id_name not in face_db:
            return jsonify({'error': 'Invalid ID'}), 400
            
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type'}), 400
            
        # Save temporary file and validate it's a valid image
        temp_path = os.path.join(IMAGES_FOLDER, f"temp_{int(time.time())}.jpg")
        file.save(temp_path)
        
        # Verify the image
        img = cv2.imread(temp_path)
        if img is None:
            os.remove(temp_path)
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Replace the original image
        dest_path = face_db[id_name]['image_path']
        shutil.move(temp_path, dest_path)
        
        if save_database():
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Failed to save database'}), 500
            
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_entry', methods=['POST'])
def delete_entry():
    """Delete an entry and its associated image"""
    try:
        data = request.json
        id_name = data.get('id')
        
        if not id_name or id_name not in face_db:
            return jsonify({'error': 'ID not found'}), 404
            
        # Delete the image file
        image_path = face_db[id_name]['image_path']
        if os.path.exists(image_path):
            os.remove(image_path)
            
        # Remove from database
        del face_db[id_name]
        
        if save_database():
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Failed to save database'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get database statistics"""
    try:
        total_entries = len(face_db)
        
        # Get backup files info
        backups = []
        if os.path.exists(BACKUP_FOLDER):
            for filename in os.listdir(BACKUP_FOLDER):
                if filename.startswith('face_embeddings_') and filename.endswith('.pkl'):
                    filepath = os.path.join(BACKUP_FOLDER, filename)
                    backups.append({
                        'filename': filename,
                        'created': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S'),
                        'size': os.path.getsize(filepath)
                    })
        
        backups.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'total_entries': total_entries,
            'backups': backups[:MAX_BACKUPS],
            'database_path': DATABASE_PATH,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(DATABASE_PATH)).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(DATABASE_PATH) else 'Never'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_backup', methods=['POST'])
def manual_backup():
    """Manually create a backup"""
    try:
        create_backup()
        return jsonify({'success': True, 'message': 'Backup created successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load database on startup
    load_database()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)