import numpy as np
import torch


def tensor_to_cuda(tensor, cuda=True, tensor_type=torch.FloatTensor):
    if not (torch.Tensor == type(tensor)):
        # print('coming here')
        tensor = torch.from_numpy(tensor * 1)
    return tensor.cuda() if cuda else tensor.cpu()
    if cuda == True:
        tensor = tensor.cuda()
    else:
        tensor = tensor.cpu()
    return tensor


def tensor_to_numpy(tensor):
    if type(tensor) == torch.Tensor:
        if tensor.device.type == "cuda":
            tensor = tensor.cpu()
        return tensor.data.numpy()
    elif type(tensor) == np.ndarray:
        return tensor
    else:
        return tensor


def copy2cpu(tensor):
    return tensor_to_numpy(tensor)


class Struct:
    def __init__(self, **kwargs):
        self.ks = []
        for key, val in kwargs.items():
            setattr(self, key, val)
            self.ks.append(key)

    def keys(
        self,
    ):
        return self.ks


c2c = copy2cpu
n2t = tensor_to_cuda
