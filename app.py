from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import base64
import pandas as pd
from PIL import Image
from utilities import *
import cv2
import requests
from io import BytesIO

app = Flask(__name__)

# Set the upload folder and allowed extensions
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"csv", "png", "jpg", "jpeg"}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/api", methods=["POST"])
def process_files():
    try:
        data = request.get_json(force=True)
        image_url = data["image_url"]

        left = int(data["left"])
        right = int(data["right"])
        top = int(data["top"])
        bottom = int(data["bottom"])
        image_file = requests.get(image_url)

        image_filename = secure_filename(os.path.basename(image_url))
        image_file_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)

        with open(image_file_path, "wb") as f:
            f.write(image_file.content)
        image = Image.open(BytesIO(image_file.content))
        im = np.array(image)
        original_image_height = im.shape[0]
        original_image_width = im.shape[1]

        im = np.array(image)
        if len(im.shape) > 2 and im.shape[2] == 4:
            im = im[:, :, :3]
        img2 = cv2.resize(im, (192, 256), interpolation=cv2.INTER_AREA)

        width = right - left
        height = bottom - top
        scaled_left = scaled_width(left, original_image_width, 192)
        scaled_top = scaled_height(top, original_image_height, 256)
        scaled_right = scaled_width(left + width, original_image_width, 192)
        scaled_bottom = scaled_height(top + height, original_image_height, 256)

        app_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the file path to the CSV file
        csv_file_path = os.path.join(app_dir, "data", "mirrorValue.csv")
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
        output_filename = "processed_" + image_filename
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        img.save(output_path)

        # Encode the processed image as a base64 string
        with open(output_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        # Return the response with the base64 encoded image and success message
        response = {
            "heatMap": encoded_image,
            "height": 0,
            "width": 0,
            "depth": 0,
        }
        return jsonify(response), 200

    except Exception as e:
        print(str(e))
        return jsonify({"error": "Server Error"}), 400


if __name__ == "__main__":
    app.run(debug=True)
