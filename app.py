from integration.supabase import supabase
from flask import Flask, request, jsonify
from flask_cors import CORS

from deepface import DeepFace
import base64
import tempfile
import os
#hello
app = Flask(__name__)
# Enable CORS for all routes and all origins
CORS(app, supports_credentials=True)
print("Starting Face Verification Service...")
@app.route("/image", methods=["POST", "OPTIONS"])
def verify_face():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

    temp_ref_path = None
    temp_captured_path = None

    try:
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
            temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_ref.write(reference_face_bytes)
            temp_ref.close()
            temp_ref_path = temp_ref.name
        except Exception as e:
            return jsonify({"error": f"Failed to download reference image: {str(e)}"}), 500

        # Decode captured image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        try:
            captured_image_bytes = base64.b64decode(image_base64)
            temp_captured = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_captured.write(captured_image_bytes)
            temp_captured.close()
            temp_captured_path = temp_captured.name
        except Exception as e:
            return jsonify({"error": f"Failed to decode captured image: {str(e)}"}), 400

        # Face verification
        try:
            result = DeepFace.verify(img1_path=temp_captured_path, img2_path=temp_ref_path)
            verified = result["verified"]
            distance = result["distance"]
            threshold = result["threshold"]

            response = {
                "success": True,
                "verified": verified,
                "face_id": face_id,
                "distance": float(distance),
                "threshold": float(threshold)
            }

            if verified:
                response["message"] = "Face verification successful! Identity confirmed."
                response["confidence"] = float(1 - (distance / threshold)) if threshold > 0 else 0
            else:
                response["message"] = "Face verification failed. Face does not match."

            final_response = jsonify(response)
            final_response.headers.add("Access-Control-Allow-Origin", "*")
            return final_response, 200

        except ValueError as e:
            final_response = jsonify({
                "success": False,
                "verified": False,
                "error": "No face detected in image(s)",
                "details": str(e)
            })
            final_response.headers.add("Access-Control-Allow-Origin", "*")
            return final_response, 400

    except Exception as e:
        final_response = jsonify({"error": str(e)})
        final_response.headers.add("Access-Control-Allow-Origin", "*")
        return final_response, 500

    finally:
        # Cleanup
        try:
            if temp_ref_path and os.path.exists(temp_ref_path):
                os.unlink(temp_ref_path)
            if temp_captured_path and os.path.exists(temp_captured_path):
                os.unlink(temp_captured_path)
        except Exception as e:
            print(f"Warning: Failed to clean up temp files: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
