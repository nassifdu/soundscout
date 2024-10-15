from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import spectrogram

# Initialize Flask app
app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Preprocess audio for model input
def process_audio(audio_data, samplerate):
    if len(audio_data) < samplerate:
        audio_data = np.pad(audio_data, (0, max(0, samplerate - len(audio_data))), mode='constant')
    _, _, spec = spectrogram(audio_data, samplerate, nperseg=min(256, len(audio_data)))
    return np.expand_dims(spec.flatten()[:44032], axis=0).astype(np.float32)

# Define a route for POST requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    audio_data, samplerate = sf.read(file)
    
    # Process and make prediction
    input_data = process_audio(audio_data, samplerate)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    predicted_label = labels[np.argmax(prediction)]
    
    return jsonify({"prediction": predicted_label})

# Run the app on your local machine
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
