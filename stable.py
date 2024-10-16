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
    required_length = 44032

    # Debugging: Log initial audio data length
    print(f"Initial audio data length: {len(audio_data)}")

    # Pad raw audio data if it is shorter than the required length
    if len(audio_data) < required_length:
        audio_data = np.pad(audio_data, (0, required_length - len(audio_data)), mode='constant')
        print(f"Audio data padded to {required_length} samples")

    # Generate the spectrogram
    _, _, spec = spectrogram(audio_data, samplerate, nperseg=min(256, len(audio_data)))
    print(f"Spectrogram shape: {spec.shape}")

    # Flatten the spectrogram and ensure it matches the required length
    flat_spec = spec.flatten()
    if len(flat_spec) < required_length:
        flat_spec = np.pad(flat_spec, (0, required_length - len(flat_spec)), mode='constant')
        print(f"Spectrogram flattened and padded to {required_length} values")

    # Normalize the spectrogram
    max_val = np.max(flat_spec)
    min_val = np.min(flat_spec)
    if max_val != min_val:  # Prevent division by zero or very small range
        flat_spec = (flat_spec - min_val) / (max_val - min_val + 1e-9)
        print("Spectrogram normalized")
    else:
        print("Warning: Spectrogram has zero variation. Skipping normalization.")

    # Return the processed audio data as a tensor input for the model
    return np.expand_dims(flat_spec, axis=0).astype(np.float32)

# Convert 3gp to mp3 using ffmpeg-python
def convert_3gp_to_mp3(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        print(f"Conversion successful! Saved as: {output_path}")
    except ffmpeg.Error as e:
        print(f"Error occurred during conversion: {e.stderr.decode()}")
        raise e

# Define a route for POST requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Save the uploaded .3gp file to a temporary file
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".3gp") as temp_3gp:
        file.save(temp_3gp.name)
        print(f"Uploaded file saved as: {temp_3gp.name}")
    
    try:
        # Convert .3gp to .mp3 using a unique temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
            mp3_path = temp_mp3.name

        # Perform the conversion
        convert_3gp_to_mp3(temp_3gp.name, mp3_path)

        # Read the converted .mp3 file using soundfile
        audio_data, samplerate = sf.read(mp3_path)
        print(f"Audio data read from mp3 file. Samplerate: {samplerate}, Length: {len(audio_data)}")

        # Process and make prediction
        input_data = process_audio(audio_data, samplerate)
        print(f"Input data shape for model: {input_data.shape}")
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        print(f"Model prediction output: {prediction}")
        
        if np.any(np.isnan(prediction)):
            print("Warning: Model returned NaN values. Check input normalization or model parameters.")
            predicted_label = "Error: Model returned NaN values"
        else:
            predicted_label = labels[np.argmax(prediction)]
            print(f"Predicted label: {predicted_label}")
    
        return jsonify({"prediction": predicted_label})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500

    finally:
        # Clean up temporary files
        os.remove(temp_3gp.name)
        print(f"Deleted temporary file: {temp_3gp.name}")
        if 'mp3_path' in locals() and os.path.exists(mp3_path):
            os.remove(mp3_path)
            print(f"Deleted temporary file: {mp3_path}")

# Run the app on your local machine
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
