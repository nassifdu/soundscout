from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import spectrogram
import tempfile
import os
import ffmpeg

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

# Convert 3gp to mp3 using ffmpeg-python
def convert_3gp_to_mp3(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        print(f"Conversion successful! Saved as: {output_path}")
    except ffmpeg.Error as e:
        print(f"Error occurred during conversion: {e.stderr.decode()}")
        raise e

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Save the uploaded .3gp file to a temporary file
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".3gp") as temp_3gp:
        file.save(temp_3gp.name)
    
    try:
        # Convert .3gp to .mp3 using a unique temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
            mp3_path = temp_mp3.name

        # Perform the conversion
        convert_3gp_to_mp3(temp_3gp.name, mp3_path)

        # Read the converted .mp3 file using soundfile
        audio_data, samplerate = sf.read(mp3_path)

        # Process and make prediction
        input_data = process_audio(audio_data, samplerate)
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        predicted_label = labels[np.argmax(prediction)]
    
        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500

    finally:
        # Clean up temporary files
        os.remove(temp_3gp.name)
        if 'mp3_path' in locals() and os.path.exists(mp3_path):
            os.remove(mp3_path)

# Run the app on your local machine
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
