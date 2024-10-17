import os
import numpy as np
import soundfile as sf
import scipy.signal
import subprocess
from flask import Flask, request, jsonify
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load TensorFlow Lite model
print("Loading TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path="soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Model loaded. Input details: {input_details}, Output details: {output_details}")

# Load labels
print("Loading labels...")
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f]
print(f"Labels loaded: {labels}")

# Function to convert audio to the required .wav format
def convert_to_wav(input_file, output_file):
    print(f"Converting {input_file} to {output_file} with ffmpeg...")
    subprocess.run([
        'ffmpeg', '-y', '-i', input_file, '-ar', '44100', '-ac', '1', output_file
    ], check=True)
    print("Conversion completed.")

# Function to preprocess the audio file for TensorFlow Lite model
def preprocess_audio(file_path):
    print(f"Preprocessing audio file {file_path}...")
    audio_data, sample_rate = sf.read(file_path)
    print(f"Original sample rate: {sample_rate}, Audio shape: {audio_data.shape}")

    if len(audio_data.shape) > 1:
        print("Converting stereo to mono...")
        audio_data = np.mean(audio_data, axis=1)

    # Ensure the sample rate is 44.1 kHz and trim/pad to 44032 samples
    target_sample_count = 44032
    current_sample_count = len(audio_data)
    
    if current_sample_count > target_sample_count:
        print(f"Trimming audio from {current_sample_count} to {target_sample_count} samples.")
        audio_data = audio_data[:target_sample_count]
    elif current_sample_count < target_sample_count:
        print(f"Padding audio from {current_sample_count} to {target_sample_count} samples.")
        padding = np.zeros(target_sample_count - current_sample_count)
        audio_data = np.concatenate((audio_data, padding))
    
    # Normalize audio data with epsilon to avoid division by zero
    epsilon = 1e-10
    max_val = np.max(np.abs(audio_data))
    audio_data = audio_data / (max_val + epsilon)
    print(f"Processed audio data: {audio_data[:10]}... (showing first 10 samples)")
    return np.array(audio_data, dtype=np.float32)

# Define POST endpoint for audio prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part in request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    input_path = "/tmp/uploaded_audio.gp3"
    wav_path = "/tmp/converted_audio.wav"
    file.save(input_path)
    print(f"File saved to {input_path}.")

    try:
        convert_to_wav(input_path, wav_path)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return jsonify({"error": f"Failed to convert file: {str(e)}"}), 500

    try:
        audio_data = preprocess_audio(wav_path)
        audio_data = np.expand_dims(audio_data, axis=0)  # Add batch dimension
        print(f"Audio data shape after expanding dimensions: {audio_data.shape}")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return jsonify({"error": f"Failed to preprocess audio: {str(e)}"}), 500

    try:
        interpreter.set_tensor(input_details[0]['index'], audio_data)
        print("Running inference...")
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Inference output: {output_data}")
    except Exception as e:
        print(f"Error during model inference: {str(e)}")
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    prediction = np.argmax(output_data)
    label = labels[prediction]
    confidence = float(output_data[0][prediction])
    print(f"Predicted label: {label} with confidence: {confidence}")

    # Clean up temporary files
    os.remove(input_path)
    os.remove(wav_path)
    print(f"Temporary files {input_path} and {wav_path} deleted.")

    return jsonify({"label": label, "confidence": confidence})

# Run Flask app on port 8080
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
