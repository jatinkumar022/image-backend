from flask import Flask, request, send_file, jsonify
from rembg import remove
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/remove-background', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    try:
        image = Image.open(file)
        output = remove(image)
        img_byte_arr = io.BytesIO()
        output.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='output.png')
    except Exception as e:
        return str(e), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
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
    else:
        return jsonify({"error": "Invalid image type"}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
