"""Small reproducible inputs and independent NumPy results for eight operator labs."""
import numpy as np


def cases(seed=17):
    rng = np.random.default_rng(seed)
    def random(*shape):
        return rng.normal(size=shape).astype(np.float32)

    x = random(8, 256)
    x[1, [0, 128]] = [3, 4]
    phase = random(8, 128)
    co, si = np.cos(phase), np.sin(phase)
    co[0], si[0], co[1, 0], si[1, 0] = 1, 0, 0, 1
    left, right = np.split(x, 2, axis=1)
    out = np.concatenate([left*co-right*si, right*co+left*si], axis=1)
    np.testing.assert_array_equal(out[1, [0, 128]], [-4, 3])
    yield "rope", (x, co, si), out

    x, gamma, bias = random(8, 128), random(128), random(128)
    x[0] = np.tile([1, 3], 64)
    gamma[:2], bias[:2], x[1] = [2, .5], [1, -1], 7
    normalized = (x-x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True)+1e-5)
    out = normalized * gamma + bias
    np.testing.assert_allclose(out[0,:2], [-.99999, -.5000025], atol=1e-6)
    np.testing.assert_array_equal(out[1], bias)
    yield "layernorm", (x, gamma, bias), out

    x, target, upstream = random(8,128), random(8,128), random(8,128)
    x[0,:7], target[0,:7], upstream[0,:7] = [-3,-2,-1,0,1,2,3], 0, [2,-1,4,7,2,3,-2]
    residual = x-target
    out = upstream*np.where(np.abs(residual)<2, residual/2, np.sign(residual))
    np.testing.assert_array_equal(out[0,:7], [-2,1,-2,0,1,3,-2])
    yield "smooth_l1_grad", (x, target, upstream), out

    image = random(4,4,128)
    image[:,:,0] = np.arange(4)[:,None]*100+np.arange(4)*10
    out = image.reshape(2,2,2,2,128).transpose(0,2,1,3,4).copy().reshape(2,2,512)
    np.testing.assert_array_equal(out[0,1,::128], [20,30,120,130])
    yield "space_to_depth", (image,), out

    table = random(8,128)
    table[0,:2], table[2,:2] = [-2,-4], [-5,-1]
    ids = rng.integers(-1,8,(8,3),dtype=np.int32)
    ids[:3] = [[2,0,2],[-1,-1,-1],[-1,0,-1]]
    out = np.zeros((8,128),np.float32)
    for bag, indices in enumerate(ids):
        valid = indices[indices>=0]
        if len(valid):
            out[bag] = table[valid].max(axis=0)
    np.testing.assert_array_equal(out[:2,:2], [[-2,-1],[0,0]])
    yield "bag_max", (table,ids), out

    slots = random(16,128)
    slots[9,:2], slots[2,:2] = [8,0], [0,4]
    routes = rng.integers(0,16,(8,2),dtype=np.int32)
    weights = rng.uniform(0,1,(8,2)).astype(np.float32)
    weights /= weights.sum(axis=1,keepdims=True)
    routes[0], weights[0] = [9,2], [.25,.75]
    out = (slots[routes]*weights[:,:,None]).sum(axis=1)
    np.testing.assert_array_equal(out[0,:2], [2,3])
    yield "moe_combine", (slots,routes,weights), out

    prob = np.tril(rng.uniform(.1,1,(8,8))).astype(np.float32)
    prob /= prob.sum(axis=1,keepdims=True)
    prob[:3,:3] = [[1,0,0],[.25,.75,0],[.5,.25,.25]]
    prob[3:,:3] = 0
    prob[3:] /= prob[3:].sum(axis=1,keepdims=True)
    grad = random(8,128)
    grad[:3,:2] = [[2,0],[0,4],[4,8]]
    out = prob.T @ grad
    np.testing.assert_array_equal(out[:3,:2], [[4,5],[1,5],[1,2]])
    yield "attention_dv", (prob,grad), out

    x, down, up, base = random(8,128), random(2,128), random(2,128), random(8,128)
    x[0] = 0
    x[0,:2], down[:,:2], up[:,:2], base[0,:2] = [2,3], [[1,-1],[2,1]], [[2,0],[1,-1]], [3,4]
    out = base + .5*((x @ down.T) @ up)
    np.testing.assert_allclose(out[0,:2], [5.5,.5])
    yield "lora", (x,down,up,base), out


if __name__ == "__main__":
    for name, inputs, expected in cases():
        print(name, [x.shape for x in inputs], "->", expected.shape)
