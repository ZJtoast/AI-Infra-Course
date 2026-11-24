"""Block-sparse left operand times a dense matrix using TPU scalar prefetch."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def sparse_mm(blocks, right, row_ptr, columns, *, interpret):
    size=128
    plan=[]
    for r,(start,end) in enumerate(zip(row_ptr,row_ptr[1:])):
        if start==end:
            plan.append((r,0,0,1,1,0))
        for i in range(start,end):
            plan.append((r,i,columns[i],i==start,i+1==end,1))
    meta=jnp.asarray(np.array(plan,np.int32).T.copy())

    def compute(schedule, left, rhs, result, acc):
        e=pl.program_id(0)
        @pl.when(schedule[3,e]!=0)
        def begin():
            acc[...]=jnp.zeros((size,size),jnp.float32)
        @pl.when(schedule[5,e]!=0)
        def multiply():
            acc[...]=acc[...]+jax.lax.dot(left[...],rhs[...],preferred_element_type=jnp.float32)
        @pl.when(schedule[4,e]!=0)
        def finish():
            result[...]=acc[...]

    def window(field):
        return pl.BlockSpec((None,size,size),lambda i,meta: (meta[field,i],0,0))
    spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1,grid=(len(plan),),
        in_specs=(window(1),window(2)),out_specs=window(0),
        scratch_shapes=(pltpu.VMEM((size,size),jnp.float32),))
    apply=pl.pallas_call(compute,grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((len(row_ptr)-1,size,size),jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),interpret=interpret)
    # Supply a valid placeholder address even if the entire sparse matrix is empty.
    if not len(blocks): blocks=jnp.zeros((1,size,size),jnp.bfloat16)
    return apply(meta,jnp.asarray(blocks,jnp.bfloat16),jnp.asarray(right,jnp.bfloat16))


if __name__=="__main__":
    I=np.eye(128,dtype=np.float32)
    blocks=np.stack([2*I,-I,3*I])
    rhs=np.stack([I,2*I,4*I])
    ptr=[0,2,2,3]
    col=[0,2,1]
    out=sparse_mm(blocks,rhs,ptr,col,interpret=jax.default_backend()=="cpu")
    np.testing.assert_array_equal(out,np.stack([-2*I,I*0,6*I]))
    rng=np.random.default_rng(29)
    lhs=rng.integers(-3,4,(3,128,128)).astype(np.float32)
    rhs=rng.integers(-2,3,(3,128,128)).astype(np.float32)
    dense=np.zeros((384,384),np.float32)
    for r in range(3):
        for i in range(ptr[r],ptr[r+1]):
            dense[r*128:(r+1)*128,col[i]*128:(col[i]+1)*128]+=lhs[i]
    out=sparse_mm(lhs,rhs,ptr,col,interpret=jax.default_backend()=="cpu")
    np.testing.assert_array_equal(out,(dense @ rhs.reshape(384,128)).reshape(3,128,128))
    out=sparse_mm([],rhs,[0,0,0],[],interpret=jax.default_backend()=="cpu")
    np.testing.assert_array_equal(out,0)
    print(jax.default_backend(),"three block rows, dense oracle and all-empty matrix passed")
