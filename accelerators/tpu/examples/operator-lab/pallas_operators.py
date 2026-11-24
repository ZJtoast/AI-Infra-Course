"""Eight standalone Pallas kernels. Run with operator_cases.py in this directory."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from operator_cases import cases


def one_tile(body, shape, inputs, interpret):
    return pl.pallas_call(body, grid=(1,), out_shape=jax.ShapeDtypeStruct(shape,jnp.float32),
                          interpret=interpret)(*inputs)


def rope(x, cosine, sine, *, interpret):
    def rotate(left, right, co, si, y0, y1):
        y0[...] = left[...] * co[...] - right[...] * si[...]
        y1[...] = right[...] * co[...] + left[...] * si[...]
    shape = jax.ShapeDtypeStruct((8,128), jnp.float32)
    run = pl.pallas_call(rotate, grid=(1,), out_shape=(shape,shape), interpret=interpret)
    y0, y1 = run(x[:,:128], x[:,128:], cosine, sine)
    return jnp.concatenate((y0,y1),axis=1)


def layernorm(x, gamma, bias, *, interpret):
    def normalize(values, gain, shift, out):
        mean = jnp.sum(values[...], axis=1, keepdims=True) / 128
        delta = values[...] - mean
        variance = jnp.sum(delta*delta, axis=1, keepdims=True) / 128
        out[...] = delta * jax.lax.rsqrt(variance+1e-5) * gain[...] + shift[...]
    return one_tile(normalize, x.shape, (x,gamma[None,:],bias[None,:]), interpret)


def smooth_l1_grad(x, target, upstream, *, interpret):
    def backward(value, goal, grad, out):
        slope = jnp.clip((value[...] - goal[...]) * .5, -1., 1.)
        out[...] = slope * grad[...]
    return one_tile(backward, x.shape, (x,target,upstream), interpret)


def space_to_depth(image, *, interpret):
    def copy_pixel(pixel, target):
        target[...] = pixel[...]
    src = pl.BlockSpec((None,None,1,128), lambda y,x,k: (2*y+k//2,2*x+k%2,0,0))
    dst = pl.BlockSpec((None,None,None,1,128), lambda y,x,k: (y,x,k,0,0))
    run = pl.pallas_call(copy_pixel, grid=(2,2,4), in_specs=(src,), out_specs=dst,
                         out_shape=jax.ShapeDtypeStruct((2,2,4,1,128),jnp.float32),
                         interpret=interpret)
    return run(image.reshape(4,4,1,128)).reshape(2,2,512)


def bag_max(table, ids, *, interpret):
    def pool(index, f0, f1, f2, out):
        row = pl.program_id(0)
        a = jnp.where(index[row,0]>=0, f0[...], -jnp.inf)
        b = jnp.where(index[row,1]>=0, f1[...], -jnp.inf)
        c = jnp.where(index[row,2]>=0, f2[...], -jnp.inf)
        any_valid = (index[row,0]>=0) | (index[row,1]>=0) | (index[row,2]>=0)
        out[...] = jnp.where(any_valid,jnp.maximum(jnp.maximum(a,b),c),0.)
    inputs = tuple(pl.BlockSpec((None,1,128), lambda r,idx,j=j: (jnp.maximum(idx[r,j],0),0,0))
                   for j in range(3))
    dst = pl.BlockSpec((None,1,128), lambda r,idx: (r,0,0))
    spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1,grid=(8,),
                                       in_specs=inputs,out_specs=dst)
    run = pl.pallas_call(pool,grid_spec=spec,out_shape=jax.ShapeDtypeStruct((8,1,128),jnp.float32),
                         interpret=interpret)
    features = table[:,None,:]
    return run(ids,features,features,features)[:,0,:]


def moe_combine(slots, routes, weights, *, interpret):
    def combine(indices, mix, a, b, out):
        r = pl.program_id(0)
        out[...] = a[...]*mix[r,0] + b[...]*mix[r,1]
    inputs = tuple(pl.BlockSpec((None,1,128),lambda r,idx,w,j=j: (idx[r,j],0,0))
                   for j in range(2))
    dst = pl.BlockSpec((None,1,128),lambda r,idx,w: (r,0,0))
    spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2,grid=(8,),in_specs=inputs,out_specs=dst)
    run = pl.pallas_call(combine,grid_spec=spec,out_shape=jax.ShapeDtypeStruct((8,1,128),jnp.float32),
                         interpret=interpret)
    x = slots[:,None,:]
    return run(routes,weights,x,x)[:,0,:]


def attention_dv(prob, grad, *, interpret):
    def accumulate(p, dy, dv):
        dv[...] = jax.lax.dot(p[...].T, dy[...], preferred_element_type=jnp.float32)
    # Pad both matrix dimensions to a full matrix tile; artificial positions carry no weight.
    p = jnp.pad(prob,((0,120),(0,120))).astype(jnp.bfloat16)
    dy = jnp.pad(grad,((0,120),(0,0))).astype(jnp.bfloat16)
    return one_tile(accumulate,(128,128),(p,dy),interpret)[:8]


def lora(x, down, up, base, *, interpret):
    def adapt(values, a0, a1, b0, b1, original, out):
        t0 = jnp.sum(values[...]*a0[...],axis=1,keepdims=True)
        t1 = jnp.sum(values[...]*a1[...],axis=1,keepdims=True)
        out[...] = original[...] + .5*(t0*b0[...] + t1*b1[...])
    return one_tile(adapt,x.shape,(x,down[:1],down[1:],up[:1],up[1:],base),interpret)


OPERATORS = (rope,layernorm,smooth_l1_grad,space_to_depth,bag_max,moe_combine,attention_dv,lora)
if __name__ == "__main__":
    cpu = jax.default_backend()=="cpu"
    for seed in (17,43):
        for kernel,(name,args,reference) in zip(OPERATORS,cases(seed)):
            assert kernel.__name__ == name
            actual = kernel(*(jnp.asarray(v) for v in args),interpret=cpu)
            if name=="attention_dv":
                # Matrix-unit operands are BF16. Compare the product of those same rounded values.
                p,g = (np.asarray(jnp.asarray(v,jnp.bfloat16),np.float32) for v in args)
                reference = p.T @ g
            np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=2e-5)
            print(jax.default_backend(),seed,name,actual.shape)
