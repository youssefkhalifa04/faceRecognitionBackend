from integration.supabase import supabase
from flask import Flask, request, jsonify

from deepface import DeepFace
from PIL import Image
import numpy as np
import io
import base64
import tempfile
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/image", methods=["POST", "OPTIONS"])
def verify_face():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    temp_ref_path = None
    temp_captured_path = None
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        face_id = data.get("face_id")
        image_base64 = data.get("image")
        
        if not face_id:
            return jsonify({"error": "face_id is required"}), 400
        
        if not image_base64:
            return jsonify({"error": "image is required"}), 400
        
        # Download reference face from Supabase storage
        try:
            reference_face_bytes = supabase.storage.from_("images").download(face_id)
            
            # Save reference face to temporary file
            temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_ref.write(reference_face_bytes)
            temp_ref.close()
            temp_ref_path = temp_ref.name
            
        except Exception as e:
            return jsonify({"error": f"Failed to download reference image: {str(e)}"}), 500
        
        # Process captured image
        # Remove data URL prefix if present (data:image/jpeg;base64,...)
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Decode base64 image
        try:
            captured_image_bytes = base64.b64decode(image_base64)
            
            # Save captured image to temporary file
            temp_captured = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_captured.write(captured_image_bytes)
            temp_captured.close()
            temp_captured_path = temp_captured.name
            
        except Exception as e:
            return jsonify({"error": f"Failed to decode captured image: {str(e)}"}), 400
        
        # Perform face verification using DeepFace
        try:
           
            result = DeepFace.verify(
                img1_path=temp_captured_path,
                img2_path=temp_ref_path,
                
            )
            

            
            verified = result["verified"]
            distance = result["distance"]
            threshold = result["threshold"]
            
            print(f"Verified: {verified}, Distance: {distance}, Threshold: {threshold}")
            
            if verified:
                return jsonify({
                    "success": True,
                    "verified": True,
                    "message": "Face verification successful! Identity confirmed.",
                    "face_id": face_id,
                    "distance": float(distance),
                    "threshold": float(threshold),
                    "confidence": float(1 - (distance / threshold)) if threshold > 0 else 0
                }), 200
            else:
                return jsonify({
                    "success": True,
                    "verified": False,
                    "message": "Face verification failed. Face does not match.",
                    "face_id": face_id,
                    "distance": float(distance),
                    "threshold": float(threshold)
                }), 200
                
        except ValueError as e:
            # No face detected in one or both images
            return jsonify({
                "success": False,
                "verified": False,
                "error": "No face detected in image(s)",
                "details": str(e)
            }), 400
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up temporary files
        try:
            if temp_ref_path and os.path.exists(temp_ref_path):
                os.unlink(temp_ref_path)
            if temp_captured_path and os.path.exists(temp_captured_path):
                os.unlink(temp_captured_path)
        except Exception as e:
            print(f"Warning: Failed to clean up temp files: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
     
