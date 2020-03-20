import numpy as np
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image

from torch_to_onnx import TorchToOnnx


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# Use PIL to manipulate images
img = Image.open("images/cat.jpg")
# Firstly, resize the image to fit the size of the model’s input (224x224).
resize = transforms.Resize([224, 224])
img = resize(img)

# Split the image into its Y, Cb, and Cr components.
# Those represent a greyscale image (Y), the blue-difference (Cb) and red-difference (Cr) chroma components.
#
# The Y component being more sensitive to the human eye is the most interested and will be transformed.
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()
# After extracting the Y component, we convert it to a tensor which will be the input of our model.
to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

# Convert super-resolution pytorch net to onnx format
net = TorchToOnnx()
torch_out, x = net.export_torch_model()

# Create onnxruntime inference session
ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

# Compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

# At this point, the output of the model is a tensor.
# Process the output of the model to construct back the final output image from the output tensor, and save the image.
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("images/cat_superres_with_ort.jpg")

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
