import numpy as np

with open("trained_model.onnx", "rb") as f:
    onnx_bytes = f.read()

# with open("trained_model.json", "r", encoding="utf-8") as f:
#     model_string = f.read()
#     model_bytes = model_string.encode("utf-8")

# with open("shap_background.pkl", "rb") as f:
#     shap_bytes = f.read()

array = np.load("trained_risks.npy")
array_bytes = array.tobytes()
array_shape = array.shape
array_dtype = str(array.dtype)

with open("assets.py", "w", encoding="raw_unicode_escape") as f:
    f.write(f"ONNX_BYTES = {repr(onnx_bytes)}\n\n")
    # f.write(f"SHAP_BYTES = {repr(shap_bytes)}\n\n")
    # f.write(f"MODEL_BYTES = {repr(model_bytes)}\n\n")
    f.write(f"ARRAY_BYTES = {repr(array_bytes)}\n")
    f.write(f"ARRAY_SHAPE = {array_shape}\n")
    f.write(f"ARRAY_DTYPE = '{array_dtype}'\n")

print("done!")