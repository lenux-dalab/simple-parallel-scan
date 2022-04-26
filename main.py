import numpy as np
import numba as nb

def serial_prefix_sum(inp_arr):
    out_arr = np.zeros(inp_arr.size)
    out_arr[0] = inp_arr[0]
    for i in nb.prange(1, inp_arr.size):
        out_arr[i] = inp_arr[i] + out_arr[i - 1]
    return out_arr

in_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
res = serial_prefix_sum(in_arr)
print(res)


@nb.njit(nopython=True, cache=False, fastmath=True, nogil=True)
def parallel_scan(a):
    # Save the input:
    b = np.copy(a)
    print(b)

    T = a.size
    bound_sweep = int(np.log2(T))
    # Up sweep
    for d in nb.prange(0, bound_sweep):
        step = 2**(d+1)
        for i in nb.prange(0, T-1, step):
            j = i + 2**d - 1
            k = i + 2**(d+1) - 1
            a[k] = a[j] + a[k]

    # 0 is actually the neutral element for
    a[T-1] = 0
    print(a)
    # Down sweep
    for d in nb.prange(bound_sweep-1, -1, -1):
        step = 2**(d+1)
        for i in nb.prange(0, T-1, step):
            j = i + 2**d - 1
            k = i + 2**(d+1) - 1
            t = a[j]
            a[j] = a[k]
            a[k] = a[k] + t
    print(a)

    # Final pass
    for i in nb.prange(0, T):
        a[i] = a[i] + b[i]

    return a

in_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
res = parallel_scan(in_arr)
print(res)


def get_neutral_element(op='+', tp='scalar'):
    if op == '+':
        if tp == 'scalar':
            return 0
        if tp == 'matrix':
            return np.identity(size)
    elif op == '*':
        if tp == 'scalar':
            return 1

def associative_op(a, b, op='+'):
    if op == '+':
        return a + b
    elif op == '*':
        return a * b
    elif op == '@':
        return a @ b

pad_power2 = lambda x, op='+' : np.pad(x, (get_neutral_element(op), int(2**np.ceil(np.log2(len(x)))) - len(x)), 'constant')

def parallel_scan2(a, op='+'):
    # Save the input:
    T = a.size
    if T & (T-1) != 0 or T == 0:
        cT = a.size
        a = pad_power2(a, op)
    else:
        cT = T

    b = np.copy(a)
    T = a.size
    bound_sweep = int(np.log2(T))
    # Up sweep
    for d in range(0, bound_sweep):
        step = 2**(d+1)
        for i in range(0, T-1, step):
            j = i + 2**d - 1
            k = i + 2**(d+1) - 1
            #a[k] = a[j] + a[k]
            a[k] = associative_op(a[j], a[k], op)

    # 0 is actually the neutral element for associative operator
    a[T-1] = get_neutral_element(op)
    # Down sweep
    for d in range(bound_sweep-1, -1, -1):
        step = 2**(d+1)
        for i in range(0, T-1, step):
            j = i + 2**d - 1
            k = i + 2**(d+1) - 1
            t = a[j]
            a[j] = a[k]
            #a[k] = a[k] + t
            a[k] = associative_op(a[k], t, op)

    # Final pass
    out = []
    for i in range(0, cT):
        #out.append(a[i] + b[i])
        out.append(associative_op(a[i], b[i], op))
    return np.array(out)

in_arr = np.array([1, 2, 3, 4, 5])
res = parallel_scan2(in_arr)
print(res)