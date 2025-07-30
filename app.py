from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="model_klakson_mini.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def home():
    return "🚗 Server Deteksi Klakson Aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if data is None or 'mfcc' not in data:
        return jsonify({'error': 'Data tidak valid'}), 400

    mfcc = np.array(data['mfcc'], dtype=np.float32).reshape(1, 13, 20, 1)
    interpreter.set_tensor(input_details[0]['index'], mfcc)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return jsonify({'klakson': bool(output > 0.5), 'confidence': float(output)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
