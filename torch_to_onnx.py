# Super Resolution model definition in PyTorch
# Super-resolution is a way of increasing the resolution of images, videos and is widely used in
# image processing or video editing

import torch.onnx
import torch.utils.model_zoo as model_zoo

from super_resolution_net import SuperResolutionNet

# Load pre-trained model weights
model_url = 'http://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1  # just a random number


class TorchToOnnx:

    def __init__(self):
        # Create the super-resolution model by using the above model definition.
        self.torch_model = SuperResolutionNet(upscale_factor=3)

    def export_torch_model(self):
        # Initialize model with the pre-trained weights
        map_location = lambda storage, loc: storage
        if torch.cuda.is_available():
            map_location = None
        self.torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

        # We must call torch_model.eval() before exporting the model, to turn it to inference mode.
        # This is required since operators like dropout or batchnorm behave differently in inference and training mode.
        self.torch_model.eval()

        # Exporting a model in PyTorch works via tracing or scripting. Here, we will use tracing.
        # To export a model, we call the torch.onnx.export() function. This will execute the model, recording a trace of
        # what operators are used to compute the outputs.
        #
        # Because export runs the model, we need to provide an input tensor x.
        # The values in this can be random as long as it is the right type and size.
        # Input size will be fixed in the exported ONNX graph for all the input’s dimensions,
        # unless specified as dynamic axes.
        #
        # In this example we export the model with an input of batch_size 1, but then specify the
        # first dimension as dynamic in the dynamic_axes parameter in torch.onnx.export().
        # The exported model will thus accept inputs of size [batch_size, 1, 224, 224] where batch_size can be variable.
        # We computed torch_out, the output after of the model, which we will use to verify that the model we exported
        # computes the same values when run in ONNX Runtime.
        x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
        torch_out = self.torch_model(x)

        # Export the model
        torch.onnx.export(self.torch_model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          "super_resolution.onnx",  # where to save the model (a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                        'output': {0: 'batch_size'}})

        return torch_out, x
