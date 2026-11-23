"""ROCm 课中 HIP 算例的 CPU 对照。运行 python reference.py，无需第三方包。"""
from itertools import accumulate
from math import exp, isclose, log, sqrt


def check(name, result, expected):
    assert len(result) == len(expected), name
    for idx, (a, b) in enumerate(zip(result, expected)):
        assert isclose(a, b, abs_tol=1e-6, rel_tol=1e-6), (name, idx, a, b)
    print(f"{name}: {result}")


def softmax(row):
    weights = [exp(x - max(row)) for x in row]
    return [v / sum(weights) for v in weights]


def conv(image, weights, stride, pad, out_size):
    result = []
    for oy in range(out_size):
        for ox in range(out_size):
            value = 0
            for ky, kernel_row in enumerate(weights):
                for kx, weight in enumerate(kernel_row):
                    iy, ix = oy * stride + ky - pad, ox * stride + kx - pad
                    if 0 <= iy < len(image) and 0 <= ix < len(image[0]):
                        value += image[iy][ix] * weight
            result.append(value)
    return result


if __name__ == "__main__":
    a = [[1, 2, 3], [4, 5, 6]]
    check("transpose", [a[r][c] for c in range(3) for r in range(2)], [1, 4, 2, 5, 3, 6])
    vector = [2, 0, -1]
    check("matvec", [sum(x*y for x, y in zip(row, vector)) for row in a], [-1, 2])
    b = [[7, 8], [9, 10], [11, 12]]
    check("gemm", [sum(x*y for x, y in zip(row,col)) for row in a for col in zip(*b)], [58,64,139,154])
    partial = [sum(range(1,5)), sum(range(5,7))]
    check("reduction tiles", partial, [10,11])
    check("reduction", [sum(partial)], [21])
    tiles = [[2,1,0,3],[4,1]]
    totals = list(accumulate(map(sum,tiles)))
    check("scan totals", totals, [6,11])
    prefix = [item+base for tile,base in zip(tiles,[0]+totals) for item in accumulate(tile)]
    check("scan", prefix, [2,3,3,6,10,11])
    image = [[1,2,3],[4,5,6],[7,8,9]]
    weights = [[-1,0],[0,1]]
    check("padded convolution", conv(image,weights,2,1,2), [1,3,7,4])
    check("valid convolution", conv(image,weights,1,0,2), [4,4,4,4])
    offsets, cols, values, x = [0,2,2,5], [0,2,0,1,2], [2,1,1,3,2], [4,5,6]
    check("CSR", [sum(values[j]*x[cols[j]] for j in range(lo,hi))
                  for lo,hi in zip(offsets,offsets[1:])], [14,0,31])
    features = [[4,1],[5,-1],[6,2]]
    check("gather", [v for i in [2,0,2] for v in features[i]], [6,2,4,1,6,2])
    row, gamma = [0,4,0,0], [1,3,1,1]
    scale = 1 / sqrt(sum(x*x for x in row)/len(row))
    check("RMSNorm", [x*scale*w for x,w in zip(row,gamma)], [0,6,0,0])
    check("softmax", softmax([0,1,2]), [.09003057,.24472847,.66524096])
    keys, values = [log(v) for v in [1,2,4,8]], [[2,0],[5,1],[8,-1],[1,3]]
    output = []
    for index, query in enumerate([1,1]):
        visible = 2 + index + 1
        prob = softmax([query*k for k in keys[:visible]])
        for col in zip(*values[:visible]):
            output.append(sum(p*v for p,v in zip(prob,col)))
    check("causal attention", output, [44/7,-2/7,52/15,22/15])
    table = [4,1,7]
    addrs = [(table[pos//2]*2+pos%2)*2+f for pos in [3,4] for f in range(2)]
    check("paged KV addresses", addrs, [6,7,28,29])
    cache = [0]*32
    for address, item in zip(addrs,[30,31,40,41]):
        cache[address] = item
    check("paged KV", [cache[i] for i in [6,7,28,29]], [30,31,40,41])
    reduced = [0,0]
    for token, gate, value in zip([0,1,0,1],[.75,.25,.25,.75],[4,6,6,7]):
        reduced[token] += gate * value
    check("MoE", reduced, [4.5,6.75])
    for width, expected in [(32,[528,1552]),(64,[2080])]:
        check(f"wave{width}", [sum(range(i+1,min(i+width,64)+1)) for i in range(0,64,width)], expected)
