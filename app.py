from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import base64
import pandas as pd
from PIL import Image
from utilities import *
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)

# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api', methods=['POST'])
def process_files():
    # Check if the request contains the required parameters
    # if 'csv_file' not in request.files or 'integer' not in request.form or 'image_file' not in request.files:
    #     return jsonify({'error': 'Missing CSV file, integer, or image file parameter'}), 400

    image_file = request.files['image_file']
    csv_file = request.files['csv_file']

    left = int(request.form['left'])
    right = int(request.form['right'])
    top = int(request.form['top'])
    bottom = int(request.form['bottom'])


    # Check if the files are valid and have the allowed extensions
    if csv_file and allowed_file(csv_file.filename) and image_file and allowed_file(image_file.filename):
        # Save the CSV file and image file to the upload folder
        csv_filename = secure_filename(csv_file.filename)
        csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        csv_file.save(csv_file_path)

        image_filename = secure_filename(image_file.filename)
        image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_file_path)

        image = Image.open(image_file)
        im = np.array(image)
        original_image_height = im.shape[0]
        original_image_width = im.shape[1]

        im = np.array(image)
        # if im has four channel i.e. including alpha channel, then remove the fourth channel
        # if im is gray scale then it has only 2 channels
        if len(im.shape) > 2 and im.shape[2] == 4:
            im = im[:, :, :3]
        img2 = cv2.resize(im, (192, 256), interpolation=cv2.INTER_AREA)

        # Read the CSV file and perform some processing
        width = right - left
        height = bottom - top
        scaled_left = scaled_width(left, original_image_width, 192)
        scaled_top = scaled_height(top, original_image_height, 256)
        scaled_right = scaled_width(left + width, original_image_width, 192)
        scaled_bottom = scaled_height(top + height, original_image_height, 256)

        df = pd.read_csv(csv_file_path)
        df_box = df.iloc[scaled_top:scaled_bottom, scaled_left:scaled_right]

        ary_box = df_box.values
        corner_choice = "Bottom Right"
        corner_dict = {
            "Top Left": 0,
            "Bottom Left": 1,
            "Bottom Right": 2,
            "Top Right": 3,
        }
        corner_value = corner_dict[corner_choice]
        dent_diff, _, _ = extract_dent(ary_box, corner_value)
        dent_heatmap, _, _ = to_heatmap(dent_diff)
        # ... Perform your desired processing on the CSV data using the provided integer ...
        contour_plot = img2.copy()
        contour_plot[scaled_top:scaled_bottom, scaled_left:scaled_right] = dent_heatmap
        img = Image.fromarray(contour_plot)

        # # Perform some processing on the image using the provided integer
        # image_filename = secure_filename(dent_heatmap.filename)
        # image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        # image_file.save(image_file_path)

        # img = Image.open(image_file_path)
        # ... Perform your desired processing on the image using the provided integer ...

        # Save the processed image
        output_filename = 'processed_' + image_filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        img.save(output_path)

        # Encode the processed image as a base64 string
        with open(output_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')

        # Return the response with the base64 encoded image and success message
        response = {
            'image': encoded_image,
            'message': 'Successful',
            'heatMap': dent_heatmap.shape
        }
        return jsonify(response), 200

    else:
        return jsonify({'error': 'Invalid CSV file, integer, or image file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)