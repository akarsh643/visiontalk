import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def get_file_size_mb(file_path):
    """Returns size of a file in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)

def convert_to_tflite(keras_model_path="models/best_model.keras", tflite_output_path="models/caption_model_quantized.tflite"):
    """Converts Keras model to dynamic range quantized TFLite model."""
    print(f"\n[INFO] Loading Keras model from: {keras_model_path}")
    model = load_model(keras_model_path, compile=False)

    print("[INFO] Initializing TFLite Converter with Dynamic Range Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Allow select TF ops if custom LSTM operations require them
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    print(f"[SUCCESS] Quantized model saved to: {tflite_output_path}")
    return tflite_output_path

def benchmark_inference(keras_model_path, tflite_model_path, num_runs=50):
    """Benchmarks execution latency between Keras and TFLite Interpreter."""
    print(f"\n[INFO] Benchmarking execution speed over {num_runs} forward passes...")

    # Load Keras model
    keras_model = load_model(keras_model_path, compile=False)

    # Load TFLite model & allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine shapes from input details
    # DenseNet feature vector input (shape: [1, 1920])
    dummy_features = np.random.randn(1, 1920).astype(np.float32)
    # Sequence input (shape: [1, max_len])
    max_len = input_details[1]["shape"][1] if len(input_details) > 1 else 34
    dummy_seq = np.random.randint(0, 100, size=(1, max_len)).astype(np.float32)

    # 1. Benchmark Keras
    start = time.time()
    for _ in range(num_runs):
        _ = keras_model.predict([dummy_features, dummy_seq], verbose=0)
    keras_latency_ms = ((time.time() - start) / num_runs) * 1000

    # 2. Benchmark TFLite
    # Map inputs correctly based on expected shapes
    feat_idx = input_details[0]["index"] if input_details[0]["shape"][1] == 1920 else input_details[1]["index"]
    seq_idx = input_details[1]["index"] if input_details[0]["shape"][1] == 1920 else input_details[0]["index"]

    start = time.time()
    for _ in range(num_runs):
        interpreter.set_tensor(feat_idx, dummy_features)
        interpreter.set_tensor(seq_idx, dummy_seq.astype(np.int32 if input_details[seq_idx]["dtype"] == np.int32 else np.float32))
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
    tflite_latency_ms = ((time.time() - start) / num_runs) * 1000

    return keras_latency_ms, tflite_latency_ms

def main():
    keras_path = "models/best_model.keras"
    tflite_path = "models/caption_model_quantized.tflite"

    if not os.path.exists(keras_path):
        print(f"[ERROR] Keras checkpoint not found at: {keras_path}")
        return

    # Convert model
    convert_to_tflite(keras_path, tflite_path)

    # Size comparison
    keras_size = get_file_size_mb(keras_path)
    tflite_size = get_file_size_mb(tflite_path)
    reduction = ((keras_size - tflite_size) / keras_size) * 100

    # Latency comparison
    try:
        keras_ms, tflite_ms = benchmark_inference(keras_path, tflite_path, num_runs=30)
        speedup = ((keras_ms - tflite_ms) / keras_ms) * 100
    except Exception as e:
        print(f"[WARNING] Latency benchmark hit a tensor alignment notice: {e}")
        keras_ms, tflite_ms, speedup = 0.0, 0.0, 0.0

    print("\n" + "=" * 50)
    print("         DAY 14: OPTIMIZATION REPORT            ")
    print("=" * 50)
    print(f"Keras Model Size   : {keras_size:.2f} MB")
    print(f"TFLite Model Size  : {tflite_size:.2f} MB")
    print(f"Storage Reduction  : {reduction:.2f}%")
    if keras_ms > 0:
        print("-" * 50)
        print(f"Keras Latency      : {keras_ms:.2f} ms / step")
        print(f"TFLite Latency     : {tflite_ms:.2f} ms / step")
        print(f"Inference Speedup  : {speedup:.2f}% faster")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()