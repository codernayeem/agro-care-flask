# Flask YOLO Prediction API

This project is a Flask-based API that performs image-based disease predictions using YOLO models for three crops: tomato, potato, and corn. The API allows users to upload images and specify the crop type to receive detailed predictions about the class and confidence level of the detected object.

## Features
- **YOLO-based Prediction**: Supports three pre-trained YOLO models for tomato, potato, and corn disease detection.
- **Image Handling**: Accepts image uploads via POST requests.
- **Dynamic Cropping**: Allows optional cropping of images before processing.
- **Detailed Results**: Provides predicted class, confidence score, and additional details based on the crop type.
- **File Storage**: Saves uploaded images in a designated `save` folder.

## Requirements
- Python 3.8+
- Flask
- Ultralytics (for YOLO models)
- PIL (for image cropping)
- NumPy

## Setup
1. Clone the repository.
   ```bash
   git clone https://github.com/codernayeem/agro-care-flask.git
   cd agro-care-flask
   ```
2. Install the dependencies (you may use a virtual environment):
   ```bash
   pip install flask ultralytics pillow numpy
   ```
3. Ensure the required files and folders exist:
   - `data.json`: Contains additional details for each crop class.
   - `model/tomato.pt`, `model/potato.pt`, `model/corn.pt`: Pre-trained YOLO model files.
   - Create a `save` folder in the root directory (if not already created).

## Usage
### Starting the Server
Run the application with:
```bash
python app.py
```
The server will start on `http://<your-local-ip>:4000`. 

The main purpose of this repository is to serve the machine learning model for the Agro Care app. The API endpoint will be stored in Firebase Cloud Firestore, from where the app will retrieve and call it. 

For more information about the app, visit the [Agro Care App repository](https://github.com/codernayeem/agro-care-app).


### API Endpoints
#### `POST /predict`
This endpoint accepts an image file and processes it using the specified model.

**Request Parameters**:
- **file** (form-data): The image file to be processed.
- **model_index** (form-data): Integer value (0 for tomato, 1 for potato, 2 for corn).
- **src** (form-data, optional): Specify the source of the image (`gallery` or `camera`). Default is `gallery`.
- **cropWidth** (form-data, optional): Specify the cropping width for the image (used if `src` is `camera`).

**Example Request**:
```bash
curl -X POST http://<your-local-ip>:4000/predict \
-F "file=@/path/to/image.jpg" \
-F "model_index=0" \
-F "src=camera" \
-F "cropWidth=300"
```

**Response**:
Returns a JSON object containing:
- `plant`: The crop type (e.g., "tomato").
- `class`: Predicted class label.
- `class_index`: Index of the predicted disease.
- `confidence`: Confidence score of the prediction.
- `details`: Additional information about the predicted disease.

## Project Structure
```
.
├── app.py             # Main Flask application
├── data.json          # Contains crop-specific class details
├── model              # Folder containing YOLO model files
│   ├── tomato.pt
│   ├── potato.pt
│   └── corn.pt
└── save               # Folder to store uploaded images
```

## Notes
- Ensure `data.json` contains appropriate mappings for the `details` field for each crop and class (disease).
- The models should be pre-trained YOLO models compatible with the `ultralytics` library.
