"""Compare the native device kernels with the independent NumPy reference."""
import ctypes
from pathlib import Path
import numpy as np
import torch
from operator_cases import cases


def main():
    if not torch.version.hip:
        raise RuntimeError("Use the ROCm build of PyTorch")
    device = "cuda"
    library = ctypes.CDLL(str(Path(__file__).with_name("operators.so")))
    launch = library.run_operator
    ptr = ctypes.c_void_p
    launch.argtypes = (ctypes.c_int, ctypes.POINTER(ptr), ptr)
    launch.restype = ctypes.c_int
    for seed in (17,43):
        for op,(name,inputs,expected) in enumerate(cases(seed)):
            data=list(inputs)
            args=[torch.from_numpy(np.ascontiguousarray(v)).to(device) for v in data]
            result=torch.full(expected.shape,float("nan"),device=device,dtype=torch.float32)
            torch.cuda.synchronize()
            addresses=(ptr*len(args))(*(v.data_ptr() for v in args))
            status=launch(op,addresses,result.data_ptr())
            if status: raise RuntimeError((name,status))
            np.testing.assert_allclose(result.cpu().numpy(),expected,rtol=2e-5,atol=2e-5)
            print(device,seed,name,expected.shape)


if __name__=="__main__":
    main()
