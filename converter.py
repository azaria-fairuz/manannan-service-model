import numpy as np

with open("trained_model.onnx", "rb") as f:
    onnx_bytes = f.read()

array = np.load("trained_risks.npy")
array_bytes = array.tobytes()
array_shape = array.shape
array_dtype = str(array.dtype)

with open("assets.py", "w") as f:
    f.write(f"ONNX_BYTES = {repr(onnx_bytes)}\n\n")
    f.write(f"ARRAY_BYTES = {repr(array_bytes)}\n")
    f.write(f"ARRAY_SHAPE = {array_shape}\n")
    f.write(f"ARRAY_DTYPE = '{array_dtype}'\n")

print("done!")