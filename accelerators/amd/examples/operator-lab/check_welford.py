"""Trace the HIP merge tree with FP32 and compare against a two-pass FP64 variance."""
import numpy as np


def merge(a,b):
    n,mean,m2=a
    k,other,rest=b
    if not n: return b
    if not k: return a
    new_n=n+k
    delta=np.float32(other-mean)
    return (new_n, np.float32(mean+delta*np.float32(k/new_n)),
            np.float32(m2+rest+delta*delta*np.float32(n*k/new_n)))


def tree(values):
    partial=[(0,np.float32(0),np.float32(0)) for _ in range(128)]
    for lane in range(128):
        for value in values[lane::128]:
            partial[lane]=merge(partial[lane],(1,np.float32(value),np.float32(0)))
    step=64
    while step:
        for lane in range(step):
            partial[lane]=merge(partial[lane],partial[lane+step])
        step//=2
    n,mean,m2=partial[0]
    return (mean,m2/n if n else 0.)


if __name__=="__main__":
    np.testing.assert_array_equal(tree([10000,10002,10004,10006]),[10003,5])
    for width in (0,1,4,129,259):
        x=10000+2*(np.arange(width)%11)
        for row in (x,np.full(width,-7)):
            ref=[row.mean(),row.var()] if width else [0,0]
            np.testing.assert_allclose(tree(row),ref,rtol=0,atol=.002)
    print("hand calculation, two rows and five widths passed")
