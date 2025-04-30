import numpy as np

class ReLU:
    """
    f(x) = {0   x <= 0
            x   x > 0}
    """
    def __call__(self, pre_activated_output):
        return np.maximum(0, pre_activated_output)

    def derivative(self, pre_activated_output, grad_so_far):
        return np.where(pre_activated_output <= 0, 0, 1) * grad_so_far

class Sigmoid:
    """
    f(x) = 1 / (1 + e^(-x))
    """
    def __call__(self, pre_activated_output):
        pre_activated_output = np.clip(pre_activated_output, -1000, 1000)   # make number less than -1000 as -1000 and number greater that 1000 as 1000 (need too handle for when x goes too big)
        sigmoid = 1 / (1 + np.exp(-pre_activated_output))
        return sigmoid

    def derivative(self, sigmoid, grad_so_far):
        return (sigmoid) * (1 - sigmoid) * grad_so_far

class Softmax:
    """
    pre_activated_output = [1,2,-3]
    softmax -> 
    exp_num = [e^1,e^2,e^-3]
    denominator = e + e^2 + e^-3
    softmax : exp_num/denominator
    (e/e + e^2 + e^-3 , e^2/e + e^2 + e^-3, e^-3/e + e^2 + e^-3)


    max_value = 2
    [-1,0,-5]
    exp_shifted = [e^-1, e^0, e^-5]
    Why we doing this?
    Case :
    [2000, 0] here calculating e^2000 is too much and cannot handeled by python so instead
    [0, -2000]
    e^0 = 1
    e^-2000 = 0
    """
    def __call__(self, pre_activated_output):
        exp_shifted = np.exp(pre_activated_output - np.max(pre_activated_output, axis=1, keepdims=True))
        denominator = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / denominator

    def derivative(self, x, grad_so_far):
        softmax_out = self.__call__(x)
        
        # Ensure grad_so_far has shape (batch_size, num_classes)
        if grad_so_far.ndim == 1:
            grad_so_far = grad_so_far.reshape(1, -1)
            
        batch_size, num_classes = softmax_out.shape
        
        # Initialize output gradient
        grad = np.zeros_like(softmax_out)
        
        for i in range(batch_size):
            s = softmax_out[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            grad[i] = np.dot(jacobian, grad_so_far[i])
            
        return grad

class BinaryStep:
    """
    f(x) = 0 if x < 0
           1 if x >= 0
    Note: Derivative is zero almost everywhere; not good for gradient-based training.
    """
    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output >= 0, 1, 0)

    def derivative(self, pre_activated_output, grad_so_far):
        return np.zeros_like(pre_activated_output)  # Derivative is zero everywhere (not usable for backprop)


class Tanh:
    """
    f(x) = tanh(x)
         = (e^x - e^-x) / (e^x + e^-x)
    derivative: 1 - tanh(x)^2
    """
    def __call__(self, pre_activated_output):
        return np.tanh(pre_activated_output)

    def derivative(self, tanh_out, grad_so_far):
        return (1 - tanh_out ** 2) * grad_so_far