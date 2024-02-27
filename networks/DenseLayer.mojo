from random import randn, rand
from tensor import TensorShape

struct DenseLayer:
    var weights: Tensor[DType.float32] 
    var biases: Tensor[DType.float32]  
    var input: Tensor[DType.float32]  

    fn __init__(inout self, input_size: Int, output_size: Int):
        self.weights = rand[DType.float32](TensorShape(output_size,input_size))
        self.biases = rand[DType.float32](TensorShape(output_size,1))

    fn forward(self, input: Tensor[DType.float32]):
        pass

    fn backward(self, output_gradient: Float32, learning_rate: Float32):
        pass