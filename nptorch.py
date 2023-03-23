#import numpy as np
import cupy as np

def sumreduce(a, b):
    while a.ndim != b.ndim:
        a = a.sum(0)
    if a.shape != b.shape:
        for idx, length in enumerate(b.shape):
            if length == 1:
                a = a.sum(idx, keepdims=True)
    return a

#----------------------------------------------------------
# implementing Torch.tensor

class Autograd:
    def __init__(self, data, children=()):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.children = set(children)
        self.grad = np.zeros_like(data)
        self.gradfn = lambda: None

    def __repr__(self):
        return (
            "Autograd"
            f" dtype:{self.data.dtype} shape:{self.data.shape}\ndata:\n{self.data}\n"
            f" grad:\n{self.grad}"
        )

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v.gradfn()

    def zero_grad(self):
        # topological order all of the children in the graph
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                v.grad = np.zeros_like(v.data)

        build_topo(self)
        
    #------------------------------------------------------------------
    # simple binary operations
    def __add__(self, other):
        other = other if isinstance(other, Autograd) else Autograd(other)
        output = Autograd(self.data + other.data, (self, other))

        def _gradfn():
            self.grad += sumreduce(output.grad, self.data)
            other.grad += sumreduce(output.grad, other.data)

        output.gradfn = _gradfn
        return output

    def __matmul__(self, other):
        other = other if isinstance(other, Autograd) else Autograd(other)
        output = Autograd(self.data @ other.data, (self, other))

        def _gradfn():
            self.grad = sumreduce(
                output.grad @ np.swapaxes(other.data, -1, -2), self.data
            )
            other.grad = sumreduce(
                np.swapaxes(self.data, -1, -2) @ output.grad, other.data
            )

        output.gradfn = _gradfn
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Autograd) else Autograd(other)
        output = Autograd(self.data * other.data, (self, other))

        def _gradfn():
            self.grad += sumreduce(other.data * output.grad, self.data)
            other.grad += sumreduce(self.data * output.grad, other.data)

        output.gradfn = _gradfn

        return output

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        output = Autograd(self.data**other, (self,))

        def _gradfn():
            self.grad += (other * self.data ** (other - 1)) * output.grad

        output.gradfn = _gradfn

        return output

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    #-------------------------------------------------------------------
    # basic operations:
    
    def __neg__(self):  # -self
        return self * -1

    def sum(self, axis=None):
        output = Autograd(np.sum(self.data, axis=axis), (self,))

        def _gradfn():
            self.grad += np.ones_like(self.data) * output.grad

        output.gradfn = _gradfn
        return output

    def reshape(self, shape):
        output = Autograd(self.data.reshape(shape), (self,))

        def _gradfn():
            self.grad += output.grad.reshape(self.data.shape)

        output.gradfn = _gradfn
        return output


    def exp(self):
        output = Autograd(np.exp(self.data), (self,))

        def _gradfn():
            self.grad += output.data * output.grad

        output.gradfn = _gradfn
        return output

    def log(self, eps=np.float32(1e-5)):
        output = Autograd(np.log(self.data + eps), (self,))

        def _gradfn():
            self.grad += (1 / (self.data + eps)) * output.grad

        output.gradfn = _gradfn
        return output

    #------------------------------------------------
    # activation functions:
    
    def sigmoid_alt(self): # numerically unstable use sigmoid instead
        return 1 / ((-self).exp() + 1)
    
    def sigmoid(self):
        a1 = self.data * (self.data >= 0)
        a2 = self.data - a1
        exp1 = np.exp(-a1)
        exp2 = np.exp(a2)
        output = Autograd((1 / (1 + exp1)) + (exp2 / (1 + exp2)) - 0.5, (self,))

        def _gradfn():
            self.grad += (1-output.data)*output.data * output.grad

        output.gradfn = _gradfn
        return output

    def softmax_alt(self): # numerically unstable
        shape = self.data.shape[1]
        return self.exp() / (self.exp() @ np.ones((shape, shape)))


    def softmax(self, dim=1): # numerically stable softmax
        e_x = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        output = Autograd(e_x / e_x.sum(axis=dim, keepdims=True), (self,))

        def _gradfn():
            self.grad += output.data * (
                output.grad
                - (output.data * output.grad).sum(axis=1, keepdims=True)
            )

        output.gradfn = _gradfn
        return output
    
    def relu(self):
        output = Autograd((self.data > 0) * self.data, (self,))

        def _gradfn():
            self.grad += (output.data > 0) * output.grad

        output.gradfn = _gradfn
        return output

    def tanh(self):
        output = Autograd(np.tanh(self.data), (self,))

        def _gradfn():
            self.grad += (1 - output.data**2) * output.grad

        output.gradfn = _gradfn
        return output


    
    #-------------------------------------------------
    # loss functions:
    
    def mse(self, other):
        assert self.data.shape == other.shape
        output = Autograd(((self.data-other)**2).mean(),(self,))
        
        def _gradfn():
            self.grad += (self.data-other)/other.size * output.grad

        output.gradfn = _gradfn
        return output

    def rmse(self, other): # use mse instead, computationally efficient 
        return (((self - other) ** 2).sum() / other.size) ** 0.5

    def crossentropy(self, other):  # numerically unstable, use logits_crossentropy instead
                                    # alternatively use softmax before using this
        return (-self.log() * other).sum()

    def logits_crossentropy(self, other, axis=1, esp=1e-8):
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        softmax = e_x / e_x.sum(axis=axis, keepdims=True)
        CEloss = (
            -np.mean(other * np.log(softmax + esp))
        )
        output = Autograd(CEloss, (self,))

        def _gradfn():
            self.grad += (softmax - other) * output.grad / other.size

        output.gradfn = _gradfn
        return output
    
    def CEsparceY(self, other, axis=1, esp=1e-8):
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        softmax = e_x / e_x.sum(axis=axis, keepdims=True)
        CEloss = (
            -np.mean(np.log(softmax + esp)[np.arange(len(other)),other])
        )
        output = Autograd(CEloss, (self,))

        def _gradfn():
            softmax[np.arange(len(other)),other]-=1
            self.grad += softmax * output.grad / other.size

        output.gradfn = _gradfn
        return output

    #-----------------------------------------------------------------------------
    # other layers

    def embedd(self, other):
        other = other if isinstance(other, Autograd) else Autograd(other)
        output = Autograd(other.data[self.data.astype("int")], (self, other))

        def _gradfn():
            dw = np.zeros_like(other.data)
            x = self.data.astype("int")
            dout = output.grad
            np.add.at(dw, x, dout)
            other.grad += dw

        output.gradfn = _gradfn
        return output
    
    def norm(self,dim=0,eps=1e-8):
        var=self.data.var(axis=dim, keepdims=True)
        mean=self.data.mean(axis=dim, keepdims=True)
        output=Autograd((self.data-mean) * (var+eps)**-0.5,(self,))
        
        def _gradfn():
            m = self.data.shape[0]
            self.grad = (
                m*output.grad \
                - output.grad.sum(axis=dim, keepdims=True) \
                - (m*output.data*(output.data*output.grad).sum(axis=dim, keepdims=True))/(m-1)
            ) / (m*(var+eps))

        output.gradfn = _gradfn
        return output
        

# END implementing Torch.tensor   
#-------------------------------------------------------------------------------------------
        
class Linear:
    def __init__(self, fan_in, fan_out, scale=1, bias=True):
        self.fan_in = fan_in
        self.fan_out = fan_out
        # k = (2/fan_in) ** 0.5
        self.scale = scale
        # self.weight = Autograd(np.random.uniform(-k, k, (fan_in, fan_out)))
        # self.bias = (
        #     Autograd(np.random.uniform(-k, k, (1, fan_out))) if bias else None
        # )
        self.weight = Autograd(np.random.randn(fan_in, fan_out)*self.scale)
        self.bias = (
            Autograd(np.random.randn(1, fan_out)*self.scale) if bias else None
        )

    def __call__(self, x):
        x = x if isinstance(x, Autograd) else Autograd(x)
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def __repr__(self):
        return (
            f"Linear({self.fan_in}, {self.fan_out},"
            f" bias={'True' if self.bias is not None else 'False'})"
        )

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class ReLU:
    def __call__(self, x):
        x = x if isinstance(x, Autograd) else Autograd(x)
        self.out = x.relu()
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Activation ReLU"


class Tanh:
    def __call__(self, x):
        x = x if isinstance(x, Autograd) else Autograd(x)
        self.out = x.tanh()
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Activation Tanh"


class Sigmoid:
    def __call__(self, x):
        x = x if isinstance(x, Autograd) else Autograd(x)
        self.out = x.sigmoid()
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Activation Signoid"


class Softmax:
    def __call__(self, x):
        x = x if isinstance(x, Autograd) else Autograd(x)
        self.out = x.softmax()
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Activation Softmax"


class MSELoss:
    def __call__(self, x, y):
        x = x if isinstance(x, Autograd) else Autograd(x)
        self.out = x.mse(y)
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Loss Mean Squared"
    
class RMSELoss:
    def __call__(self, x, y):
        x = x if isinstance(x, Autograd) else Autograd(x)
        self.out = x.rmse(y)
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Loss Root Mean Squared"

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def __repr__(self):
        return (
            "==============================\n"
            "       SEQUENTIAL MODEL       \n"
            "==============================\n"
            f"Total params = {self.total_params()}\n"
            f"Layers = {self.layers}\n"
            "==============================\n"
        )

    def train_mode(self, mode=True):
        for layer in self.layers:
            layer.training = mode

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def total_params(self):
        size = 0
        for params in self.parameters():
            size += params.data.size
        return size


class Dropout:
    def __init__(self, p=0.5):
        self.training = True
        self.p = p
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be a probability")

    def __call__(self, x):
        x = x if isinstance(x, Autograd) else Autograd(x)
        if self.training:
            shape = x.data.shape
            self.out = x * (np.random.binomial(size=shape, n=1, p=1 - self.p).astype("float"))/ self.p
        else:
            self.out = x
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"
    
    
class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.training = True
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = Autograd(
            np.random.randn(self.vocab_size, self.embedding_dim)
        )

    def __call__(self, x):
        x = x if isinstance(x, Autograd) else Autograd(x)
        shape = x.data.shape
        self.out = x.embedd(self.weight)
        return self.out

    def parameters(self):
        return [self.weight]

    def __repr__(self):
        return (
            f"Embedding(vocab_size={self.vocab_size},"
            f" embedding_dim={self.embedding_dim})"
        )
    
class BatchNorm:
    def __init__(self, n_features, eps=1e-5, momentum=0.1, dim=0):
        self.eps = eps
        self.momentum = momentum
        self.dim = dim
        self.training = True
        self.gamma = Autograd(np.ones(n_features))
        self.beta = Autograd(np.zeros(n_features))
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
  
    def __call__(self, x):
        if self.training:
            if x.data.ndim == 2:
                self.dim = 0
            elif x.data.ndim == 3:
                self.dim = (0,1)
            xmean = x.data.mean(self.dim, keepdims=True) # batch mean
            xvar = x.data.var(self.dim, keepdims=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
            
        self.out = self.gamma * x.norm(dim=self.dim,eps=1e-8) + self.beta

        if self.training:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
  
    def parameters(self):
        return [self.gamma, self.beta]

class FlattenConsecutive:
  
    def __init__(self, n):
        self.n = n
    
    def __call__(self, x):
        B, T, C = x.data.shape
        T, C = T//self.n, C*self.n
        x = x.reshape((B, T, C))
        if x.data.shape[1] == 1:
            x = x.reshape((B, C))
        self.out = x
        return self.out

    def parameters(self):
        return []
    
class Reshape:
  
    def __init__(self, new_shape):
        self.new_shape = new_shape
    
    def __call__(self, x):
        self.out = x.reshape(self.new_shape)
        return self.out

    def parameters(self):
        return []
# class LayerNorm:
#     def __init__(self, eps=1e-5, momentum=0.1, normalized_shape=1,dim=1):
#         self.eps = eps
#         self.momentum = momentum
#         self.dim = dim
#         self.training = True
#         self.gamma = Autograd(np.ones(n_features))
#         self.beta = Autograd(np.zeros(n_features))
  
#     def __call__(self, x):
#         if self.training:
  
#         self.out = self.gamma * x.norm(dim=self.dim,eps=1e-8) + self.beta

        
#         return self.out
  
#     def parameters(self):
#         return [self.gamma, self.beta]
