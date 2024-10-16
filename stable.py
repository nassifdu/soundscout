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
    print("--- Preprocessing Audio ---")
    print(f"Audio data length: {len(audio_data)}")
    print(f"Samplerate: {samplerate}")

    if len(audio_data) < samplerate:
        print("Padding audio data to match samplerate...")
        audio_data = np.pad(audio_data, (0, max(0, samplerate - len(audio_data))), mode='constant')
        print(f"Padded audio data length: {len(audio_data)}")
    else:
        print("Audio data length is sufficient, no padding needed.")

    # Generate the spectrogram
    _, _, spec = spectrogram(audio_data, samplerate, nperseg=min(256, len(audio_data)))
    print(f"Spectrogram shape: {spec.shape}")

    # Flatten the spectrogram to match model input
    flat_spec = spec.flatten()
    if len(flat_spec) < 44032:
        print("Padding flattened spectrogram...")
        flat_spec = np.pad(flat_spec, (0, 44032 - len(flat_spec)), mode='constant')
    elif len(flat_spec) > 44032:
        print("Trimming flattened spectrogram...")
        flat_spec = flat_spec[:44032]
    print(f"Final flattened spectrogram length: {len(flat_spec)}")

    return np.expand_dims(flat_spec, axis=0).astype(np.float32)

# Define a route for POST requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("Error: No file provided in the request.")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    print(f"Received file: {file.filename}")
    
    try:
        # Read the audio data from the file
        audio_data, samplerate = sf.read(file)
        print(f"Audio data read. Length: {len(audio_data)}, Samplerate: {samplerate}")

        # Process the audio and make prediction
        input_data = process_audio(audio_data, samplerate)
        print(f"Input data shape: {input_data.shape}")
        
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        print(f"Model prediction output: {prediction}")

        predicted_label = labels[np.argmax(prediction)]
        print(f"Predicted label: {predicted_label}")
    
        return jsonify({"prediction": predicted_label})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500

# Run the app on your local machine
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8080)
