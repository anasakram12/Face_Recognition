from flask import Flask, request, jsonify
import psycopg2
from datetime import datetime

app = Flask(__name__)

# PostgreSQL connection details
DB_HOST = "localhost"
DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "ballyssql"

# Function to connect to PostgreSQL
def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# API Route to Insert Data
@app.route('/add_log', methods=['POST'])
def add_log():
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Invalid input format. Expected a JSON array of log entries."}), 400

    try:
        conn = connect_db()
        cur = conn.cursor()
        sql = """
        INSERT INTO log_entries (logDate, status, inputFilename, inputPath, outputFilename, outputPath, confidence, matchedFilename, camera_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for entry in data:
            if not isinstance(entry, dict):
                return jsonify({"error": "Invalid log entry format. Each entry must be a JSON object."}), 400
            
            values = (
                entry.get("logDate"),
                entry.get("status"),
                entry.get("inputFilename"),
                entry.get("inputPath"),
                entry.get("outputFilename"),
                entry.get("outputPath"),
                entry.get("confidence"),
                entry.get("matchedFilename"),
                entry.get("camera_id")  
            )
            cur.execute(sql, values)
        
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "Log entries added successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get logs within a time range
@app.route('/get_logs', methods=['GET'])
def get_logs():
    try:
        time_from = request.args.get('from')
        time_to = request.args.get('to')

        if not time_from or not time_to:
            return jsonify({"error": "Both 'from' and 'to' parameters are required"}), 400

        try:
            datetime.strptime(time_from, '%Y-%m-%d %H:%M:%S')
            datetime.strptime(time_to, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use 'YYYY-MM-DD HH:MM:SS'"}), 400

        conn = connect_db()
        cur = conn.cursor()

        sql = """
        SELECT id, logDate, inputFilename, matchedFilename, confidence, camera_id
        FROM log_entries
        WHERE logDate BETWEEN %s AND %s
        ORDER BY logDate DESC
        """
        
        cur.execute(sql, (time_from, time_to))
        rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],  # Explicitly include id
                "logDate": row[1].strftime('%Y-%m-%d %H:%M:%S') if row[1] else None,
                "inputFilename": row[2],
                "matchedFilename": row[3],
                "confidence": float(row[4]) if row[4] is not None else None,
                "camera_id": row[5]
            })

        cur.close()
        conn.close()

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get unrecognized saved logs within a time range
@app.route('/get_unrecognized_logs', methods=['GET'])
def get_unrecognized_logs():
    try:
        time_from = request.args.get('from')
        time_to = request.args.get('to')

        if not time_from or not time_to:
            return jsonify({"error": "Both 'from' and 'to' parameters are required"}), 400

        try:
            datetime.strptime(time_from, '%Y-%m-%d %H:%M:%S')
            datetime.strptime(time_to, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use 'YYYY-MM-DD HH:MM:SS'"}), 400

        conn = connect_db()
        cur = conn.cursor()

        sql = """
        SELECT id, inputFilename
        FROM log_entries
        WHERE status = 'unrecognized_saved'
        AND logDate BETWEEN %s AND %s
        ORDER BY logDate DESC
        """
        
        cur.execute(sql, (time_from, time_to))
        rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "inputFilename": row[1]
            })

        cur.close()
        conn.close()

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to update a log entry
@app.route('/update_log', methods=['POST'])
def update_log():
    try:
        data = request.get_json()
        
        required_fields = ['id', 'status', 'confidence', 'matchedFilename']
        if not all(field in data for field in required_fields):
            return jsonify({
                "error": "Missing required fields. Need id, status, confidence, and matchedFilename"
            }), 400

        conn = connect_db()
        cur = conn.cursor()

        sql = """
        UPDATE log_entries 
        SET status = %s, 
            confidence = %s, 
            matchedFilename = %s
        WHERE id = %s
        RETURNING id
        """
        
        cur.execute(sql, (
            data['status'],
            data['confidence'],
            data['matchedFilename'],
            data['id']
        ))
        
        updated_row = cur.fetchone()
        conn.commit()
        
        cur.close()
        conn.close()
        
        if updated_row:
            return jsonify({
                "message": "Log entry updated successfully",
                "id": updated_row[0]
            }), 200
        else:
            return jsonify({
                "error": f"No log entry found with id {data['id']}"
            }), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)