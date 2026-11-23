"""Recompute the chapter's numbers on a CPU with only the Python standard library."""
from itertools import accumulate
from math import ceil, exp, floor, isclose, log, sqrt


def close(xs, ys):
    assert len(xs) == len(ys)
    assert all(isclose(x,y,rel_tol=1e-6,abs_tol=1e-6) for x,y in zip(xs,ys)), (xs,ys)


def mfma_distribution():
    M,N,K=17,35,5
    a=[[r+k+1 for k in range(K)] for r in range(M)]
    b=[[(k+1)*(1+c%3) for c in range(N)] for k in range(K)]
    dense=[[sum(v*b[k][c] for k,v in enumerate(row)) for c in range(N)] for row in a]
    mapped=[[None]*N for _ in range(M)]
    for r0 in range(0,M,16):
        for c0 in range(0,N,32):
            for wave in range(2):
                regs=[[0]*4 for _ in range(64)]
                for k0 in range(0,K,4):
                    av,bv=[],[]
                    for lane in range(64):
                        x,y=lane%16,lane//16
                        row,col=r0+x,c0+wave*16+x
                        av.append(a[row][k0+y] if row<M and k0+y<K else 0)
                        bv.append(b[k0+y][col] if col<N and k0+y<K else 0)
                    for lane in range(64):
                        x,y=lane%16,lane//16
                        for reg in range(4):
                            row=y*4+reg
                            regs[lane][reg]+=sum(av[k*16+row]*bv[k*16+x] for k in range(4))
                for lane in range(64):
                    for reg in range(4):
                        row=r0+4*(lane//16)+reg
                        col=c0+16*wave+lane%16
                        if row<M and col<N: mapped[row][col]=regs[lane][reg]
    assert dense==mapped and mapped[4][17]==345
    return "17x35 checked, C[4,17]=345; inputs 256 vs 192 words/full tile"


def row_attention():
    weights=[1,3,2,6,12]
    v=[[2,0,1,-1],[8,1,0,2],[-1,2,3,0],[3,-1,2,4],[4,0,0,1]]
    first=[]
    for n in (3,4,5):
        expected=[sum(weights[j]*v[j][f] for j in range(n))/sum(weights[:n]) for f in range(4)]
        m,den,num=-float("inf"),0,[0]*4
        for start in range(0,n,4):
            js=range(start,min(start+4,n))
            new_m=max(m,max(log(weights[j]) for j in js))
            keep=exp(m-new_m)
            w=[exp(log(weights[j])-new_m) for j in js]
            den=den*keep+sum(w)
            num=[total*keep+sum(p*v[j][f] for j,p in zip(js,w)) for f,total in enumerate(num)]
            m=new_m
        close([z/den for z in num],expected);first.append(expected[0])
    close(first,[4,3.5,3.75])
    return first


def group_quant():
    x=[254,-63,1,-1,-81]+[0]*59+[63.5,-5.25]
    scales,codes,restored=[],[],[]
    for row in (x,[0]*66):
        for i in range(0,len(row),64):
            tile=row[i:i+64]
            s=max(map(abs,tile))/127 or 1
            q=[(-1 if v<0 else 1)*floor(abs(v/s)+.5) for v in tile]
            y=[s*v for v in q]
            assert all(abs(a-b)<=s/2+1e-7 for a,b in zip(y,tile))
            scales.append(s);codes+=q;restored+=y
    assert scales==[2,.5,1,1] and codes[:5]==[127,-32,1,-1,-41] and codes[64:66]==[127,-11]
    return scales,codes[:5],restored[64:66]


def route_packing():
    for tags in ([2,0,2,1,0,2,1,1],[(i*7+i//3)%3 for i in range(70)]):
        tiles=[tags[i:i+64] for i in range(0,len(tags),64)]
        masks=[[sum(1<<i for i,v in enumerate(tile) if v==e) for e in range(3)] for tile in tiles]
        dest=[None]*len(tags);pos=0
        for e in range(3):
            for tile,keys in enumerate(tiles):
                mask=masks[tile][e]
                for lane,key in enumerate(keys):
                    if key==e: dest[pos+(mask & ((1<<lane)-1)).bit_count()]=tile*64+lane
                pos+=mask.bit_count()
        assert dest==sorted(range(len(tags)),key=tags.__getitem__)
    return sorted(range(8),key=[2,0,2,1,0,2,1,1].__getitem__)


def scan_and_banks():
    x=[2,1,0,3,4,1,0,2]
    tree=x.copy();d=1
    while d<len(x):
        for j in range(2*d-1,len(x),2*d): tree[j]+=tree[j-d]
        d*=2
    tree[-1]=0;d//=2
    while d:
        for j in range(2*d-1,len(x),2*d):tree[j-d],tree[j]=tree[j],tree[j]+tree[j-d]
        d//=2
    prefix=[u+v for u,v in zip(tree,x)]
    assert prefix==list(accumulate(x))
    assert len(set((lane*32)%32 for lane in range(32)))==1
    assert len(set((lane*33)%32 for lane in range(32)))==32
    assert 34*35+1==1191 and 37+34==71 and 64*4-9*9==175
    return prefix, "one vs 32 distinct banks for a column's 32 scalar addresses"


def original_kernels():
    x=[2,1,0,-1];r=[-2,3,0,1];u=[a+b for a,b in zip(x,r)]
    close([v*w/sqrt(sum(k*k for k in u)/4) for v,w in zip(u,[1,3,1,1])],[0,6,0,0])
    image=[[2,1,3],[3,2,8],[5,6,1]]
    conv=[image[r+1][c+1]-image[r][c] for r in range(2) for c in range(2)]
    assert conv==[0,7,3,-1] and max(conv)==7
    ptr,cols,w=[0,2,2,5],[0,2,0,1,2],[2,1,1,3,2]
    features=[[4,1,0,2],[5,-1,2,0],[6,2,1,-2]]
    y=[[sum(w[e]*features[cols[e]][f] for e in range(ptr[r],ptr[r+1])) for f in range(4)] for r in range(3)]
    assert y==[[14,4,1,2],[0,0,0,0],[31,2,8,-2]]
    page=[[(30,31),(99,99)],[(40,41),(50,51)],[(10,11),(20,21)]]
    table=[[2,0],[2,1]]
    values=[[page[table[r][j//2]][j%2] if j<n else (0,0) for j in range(4)] for r,n in enumerate((3,4))]
    assert values[0][-1]==(0,0) and values[1][-1]==(50,51)
    return conv,y,values


if __name__=="__main__":
    for f in [mfma_distribution,row_attention,group_quant,route_packing,scan_and_banks,original_kernels]:
        print(f.__name__,f())
