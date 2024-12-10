from flask import Flask, jsonify
import socket
import ultralytics
from flask import request
import numpy as np
import datetime
import json

app = Flask(__name__)

all_data = json.load(open("data.json", "r"))

model_0_path = "model/tomato.pt"
model_1_path = "model/potato.pt"
model_2_path = "model/corn.pt"

model_0 = ultralytics.YOLO(model_0_path)
model_1 = ultralytics.YOLO(model_1_path)
model_2 = ultralytics.YOLO(model_2_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model_index' not in request.form:
        return jsonify({"error": "No file or model index provided"}), 400

    file = request.files['file']
    model_index = int(request.form['model_index'])

    if model_index not in [0, 1, 2]:
        return jsonify({"error": "Invalid model index"}), 400
    
    # save file to "save"
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"save\\save_{timestamp}.jpg"
    file.save(file_path)

    if model_index == 0:
        model = model_0
        plant = "tomato"
    elif model_index == 1:
        model = model_1
        plant = "potato"
    else:
        model = model_2
        plant = "corn"
    
    results = model.predict(file_path, verbose=False, save=False, plots=False)
    class_prob = results[0].probs.data.numpy()
    predicted_class_index = np.argmax(class_prob)
    classes = model.model.names
    predicted_class = classes[predicted_class_index]
    confidence = class_prob[predicted_class_index]

    print("==================================================")
    print(predicted_class_index, confidence, predicted_class)
    print("==================================================")

    details = all_data[f"details_{plant}"][predicted_class]

    return jsonify({"plant": plant, "class": predicted_class, "class_index": int(predicted_class_index), "confidence": round(float(confidence), 4), "details": details})

if __name__ == '__main__':
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    app.run(host=local_ip, port=4000, debug=True)