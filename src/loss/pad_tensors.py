import torch


def pad_tensors_to_match(tensor1, tensor2, padding_value=0):
    if tensor1.dim() != tensor2.dim():
        raise ValueError("The number of dimensions does not match")

    max_shape = [max(s1, s2) for s1, s2 in zip(tensor1.shape, tensor2.shape)]

    def pad_tensor(tensor, target_shape):
        padding = []
        for i in range(len(target_shape) - 1, -1, -1):
            pad_size = target_shape[i] - tensor.size(i)
            padding.extend([0, pad_size])
        return torch.nn.functional.pad(tensor, padding, value=padding_value)

    padded_tensor1 = pad_tensor(tensor1, max_shape)
    padded_tensor2 = pad_tensor(tensor2, max_shape)
    return padded_tensor1, padded_tensor2
