from flask import Flask, request, send_file, jsonify
from flask_cors import CORS, cross_origin
from rembg import remove
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import numpy as np
import cv2
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins; change as needed

# Setup logging
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health')
def health_check():
    return "OK", 200


@app.route('/remove-background', methods=['POST'])
@cross_origin(origins='*')
def remove_background():
    if 'image' not in request.files:
        logging.info("No file part")
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    if file.filename == '':
        logging.info("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        logging.info("Invalid file type")
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Open the image file
        image = Image.open(file)
        
        # Process the image to remove the background
        output = remove(image)

        # Save the output to a file
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'background_removed_image.png')
        output.save(output_path, format='PNG')
        
        # Return the URL for the processed image
        return jsonify({
            "processed_image_url": f'/download/background_removed_image'
        })
    except Exception as e:
        logging.error(f"Error during background removal: {e}")
        return jsonify({"error": "Error during background removal"}), 500




@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    img = Image.open(file.stream)
    original_img_path = os.path.join(UPLOAD_FOLDER, 'original_image.jpg')
    img.save(original_img_path)

    open_cv_image = np.array(img)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    pil_img = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    contrast_enhancer = ImageEnhance.Contrast(brightness_enhancer.enhance(1))
    enhanced_image = contrast_enhancer.enhance(1.1)
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
    sharpened_image = sharpness_enhancer.enhance(2.0)
    smooth_image = enhanced_image.filter(ImageFilter.SMOOTH_MORE)

    enhanced_image_cv = np.array(enhanced_image)
    enhanced_image_cv = cv2.cvtColor(enhanced_image_cv, cv2.COLOR_RGB2BGR)

    scale_factor = 3
    width = int(enhanced_image_cv.shape[1] * scale_factor)
    height = int(enhanced_image_cv.shape[0] * scale_factor)
    resized_image = cv2.resize(enhanced_image_cv, (width, height), interpolation=cv2.INTER_CUBIC)

    smoothed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    sharpening_kernel = np.array([[0, -0.1, 0],
                                  [-0.1, 1.5, -0.1],
                                  [0, -0.1, 0]])
    sharpened_image = cv2.filter2D(smoothed_image, -1, sharpening_kernel)

    processed_img_path = os.path.join(UPLOAD_FOLDER, 'processed_image.jpg')
    cv2.imwrite(processed_img_path, sharpened_image)

    return jsonify({
        "processed_image_url": '/download/processed_image'
    })

@app.route('/download/<image_type>', methods=['GET'])
def download_image(image_type):
    if image_type == "original_image":
        return send_file(os.path.join(UPLOAD_FOLDER, 'original_image.jpg'), as_attachment=True)
    elif image_type == "processed_image":
        return send_file(os.path.join(UPLOAD_FOLDER, 'processed_image.jpg'), as_attachment=True)
    elif image_type == "background_removed_image":
        return send_file(os.path.join(UPLOAD_FOLDER, 'background_removed_image.png'), as_attachment=True)
    else:
        return jsonify({"error": "Invalid image type"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
