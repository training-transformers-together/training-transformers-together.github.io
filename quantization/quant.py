# Quantization reduces a bit representation to less bits for efficient storage or computation.
# Most floating point data types have a mapping from a bit representation, e.g. 0010 = 2 to a floating
# point representation 2 -> 2 / max(0010) = 2/15 = 0.133333
# As such, we can represent a floating point quantization a mapping from integers to floating point values, e.g.
# [0, 1, 2, 3] -> [-1.0, -0.25, 0.25 , 1.0]
import numpy as np
from scipy.spatial.distance import cdist

index = np.array([0, 1, 2, 3, 4, 5, 6, 7])
values = np.linspace(-1.0, 1.0, 8) # 3-bit linear quantization
print('quantization values:', values)

# To quantize an input distribution we first need to normalize its range into the range of the quantization values, in this case [-1.0, 1.0]
# We can do this through division by the abolute maximum value if our distribution is roughly symmetric (most distribution in deep learning are noramlly distributed)

rand_inputs = np.random.randn(1024, 1024).astype(np.float32)

absmax = np.max(np.abs(rand_inputs))
normed = rand_inputs / absmax
print('normalized min and max range', np.min(normed), np.max(normed))

# The next step is to round the input value to the closest quantization value. 
# This can be done by performing a binary search of each element of the normalized input tensor with respect to the sorted values array:
# In this case, we simply compute the distance between all values and find the closest directly.

dist = cdist(normed.flatten().reshape(-1, 1), values.reshape(-1, 1))
closest_idx = np.argmin(dist, 1).reshape(rand_inputs.shape)

val, count = np.unique(closest_idx, return_counts=True)
print('Values:', val)
print('Count:', count)

# Closest index now represents the quantized 3 bit representation (4 different values). We can use this representation to store the data efficiently.


# ==================DEQUANTIZATION========================
# To dequantize the tensor we reverse the operations the we did
# 1. lookup the values corresponding to the 3-bit index
# 2. Denormalize by multipying by absmax

dequant = values[closest_idx]*absmax
# mean absolute error:
error = np.abs(dequant-rand_inputs).mean()
print(f'Absolute linear 3-bit quantization error: {error:.4f}')

# This yields an error of about 0.34 per value. We can do better with non-linear quantization.

# ==================NON-LINEAR QUANTIZATION========================
# In non-linear quantization the distance between quantization values is not always equal.
# This allows us to allocate more values to regions of high density. For example, the normal distribution has many values around 0.
# This can reduce the overall error in the distribution.
index = np.array([0, 1, 2, 3, 4, 5, 6, 7])
values = np.array([-1.0, -0.5, -0.25, -0.075, 0.075, 0.25, 0.5, 1.0])

dist = cdist(normed.flatten().reshape(-1, 1), values.reshape(-1, 1))
closest_idx = np.argmin(dist, 1).reshape(rand_inputs.shape)

val, count = np.unique(closest_idx, return_counts=True)
print('Values:', val)
print('Count:', count)

dequant = values[closest_idx]*absmax
error = np.abs(dequant-rand_inputs).mean()
print(f'Absolute non-linear 3-bit quantization error: {error:.4f}')

# dynamic quantization
# Adaptive from: https://github.com/facebookresearch/bitsandbytes/blob/main/bitsandbytes/functional.py
def create_dynamic_map(signed=True, n=7):
    '''
    Creates the dynamic quantiztion map.
    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.
    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.
    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    '''

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    additional_items = 2**(7-n)-1
    if not signed: additional_items = 2*additional_items
    for i in range(n):
        fraction_items = 2**(i+7-n)+1 if signed else 2**(i+7-n+1)+1
        boundaries = np.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1]+boundaries[1:])/2.0
        data += ((10**(-(n-1)+i))*means).tolist()
        if signed:
            data += (-(10**(-(n-1)+i))*means).tolist()

    if additional_items > 0:
        boundaries = np.linspace(0.1, 1, additional_items+1)
        means = (boundaries[:-1]+boundaries[1:])/2.0
        data += ((10**(-(n-1)+i))*means).tolist()
        if signed:
            data += (-(10**(-(n-1)+i))*means).tolist()

    data.append(0)
    data.append(1.0)
    data.sort()
    return np.array(data)

import time

values = create_dynamic_map(signed=True)

t0 = time.time()
dist = cdist(normed.flatten().reshape(-1, 1), values.reshape(-1, 1))
closest_idx = np.argmin(dist, 1).reshape(rand_inputs.shape)
quant_time = time.time()-t0

dequant = values[closest_idx]*absmax
error = np.abs(dequant-rand_inputs).mean()
print(f'Absolute dynamic 8-bit quantization error: {error:.4f}')
print(f'Total time taken: {quant_time:.4f} seconds.')

# This yields an error as low as 0.012. We could do even better when we use block-wise quantization.
# But performing block-wise quantization without optimized code is a bit slow. We can use the bitsandbytes library to do this quickly.

import torch
import bitsandbytes.functional as F

rand_inputs = torch.from_numpy(rand_inputs)
t0 = time.time()
quant_values, quant_state = F.quantize_blockwise(rand_inputs)
quant_time = time.time()-t0
dequant_values = F.dequantize_blockwise(quant_values, quant_state)

error = torch.abs(dequant_values-rand_inputs).mean().item()
print(f'Absolute dynamic block-wise 8-bit quantization error: {error:.4f}')
print(f'Total time taken (CPU): {quant_time:.4f} seconds.')

rand_inputs = rand_inputs.cuda()
t0 = time.time()
quant_values, quant_state = F.quantize_blockwise(rand_inputs)
quant_time = time.time()-t0
print(f'Total time taken (GPU): {quant_time:.4f} seconds.')


