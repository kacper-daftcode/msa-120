"""Verify: real-checkpoint weight NVFP4 GEMM via group_gemm matches dequant ref.
Tests the on-disk linear E4M3 scale -> swizzle -> mma path for ONE real expert."""
import json, torch, flashinfer
from safetensors import safe_open
from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

ROOT="/models/MiniMax-M3-NVFP4"
FP4_MAX=6.0; E4M3_MAX=448.0
idx=json.load(open(f"{ROOT}/model.safetensors.index.json"))['weight_map']
_h={}
def get(key):
    f=idx[key]
    if f not in _h: _h[f]=safe_open(f"{ROOT}/{f}",framework="pt",device="cpu")
    return _h[f].get_tensor(key)

def rel_rms(a,b):
    a=a.float();b=b.float();return ((a-b).pow(2).mean().sqrt()/(b.pow(2).mean().sqrt()+1e-20)).item()

# E2M1 LUT (uint8 nibble -> value)
_E2M1=torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,
                    -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],dtype=torch.float32)
def unpack_e2m1(packed):  # [N, K//2] uint8 -> [N, K] float
    lo=packed & 0x0F; hi=(packed>>4)&0x0F
    out=torch.empty(packed.shape[0],packed.shape[1]*2,dtype=torch.float32)
    out[:,0::2]=_E2M1[lo.long()]; out[:,1::2]=_E2M1[hi.long()]
    return out

def dequant_weight(w_packed, w_scale_e4m3, w_scale_2):
    # w = e2m1 * (block_scale_E4M3) * global; block scale per 16 cols
    N,Khalf=w_packed.shape; K=Khalf*2
    q=unpack_e2m1(w_packed)                      # [N,K]
    bs=w_scale_e4m3.float()                       # [N,K//16]
    eff=(bs*w_scale_2.float()).repeat_interleave(16,dim=1)  # [N,K]
    return q*eff

def swizzle_to_mma(w_scale_e4m3, N, K):
    # on-disk linear [N, K//16] E4M3 -> swizzle (pad m to 128) -> mma layout
    sf=w_scale_e4m3.view(torch.uint8).cuda()      # reinterpret e4m3 bytes as uint8
    Npad=((N+127)//128)*128
    if N<Npad:
        sf=torch.cat([sf,torch.zeros(Npad-N,sf.shape[1],dtype=torch.uint8,device=sf.device)],0)
    sf_sw=flashinfer.nvfp4_block_scale_interleave(sf)   # swizzled, same shape
    return convert_sf_to_mma_layout(sf_sw, m=Npad, k=K, num_groups=1)

def quant_gs(t): return (FP4_MAX*E4M3_MAX)/t.abs().amax().float().clamp_min(1e-12)
def quant_a(t,gs): return flashinfer.fp4_quantize(t,global_scale=gs,sf_vec_size=16,is_sf_swizzled_layout=True)

L=10; E=0
base=f"model.language_model.layers.{L}.mlp.experts.{E}.gate_proj"
w_packed=get(f"{base}.weight").cuda()            # [I, H//2] uint8
w_scale=get(f"{base}.weight_scale").cuda()        # [I, H//16] e4m3
w_scale_2=get(f"{base}.weight_scale_2").cuda()    # scalar
N,Khalf=w_packed.shape; K=Khalf*2
print(f"gate_proj: N(I)={N} K(H)={K}")

w_bf16=dequant_weight(w_packed.cpu(),w_scale.cpu(),w_scale_2.cpu()).cuda().bfloat16()

torch.manual_seed(0)
M=64
a=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
a_gs=quant_gs(a); a_q,a_sf=quant_a(a,a_gs)
b_gs_q=1.0/w_scale_2.float()   # dequant uses w_scale_2 as global; quant gs = 1/global? 
# alpha needs 1/(a_gs * w_global). w dequant multiplier already folds w_scale_2.
# But b_q here are the RAW packed e2m1 (no extra global). The block scale (w_scale)
# does NOT include global; global is w_scale_2. So effective b dequant = e2m1*w_scale*w_scale_2.
# group_gemm computes: alpha * sum(a_e2m1*a_blockscale * b_e2m1*b_blockscale).
# a_blockscale absorbs 1/a_gs implicitly? No: a_scale (e4m3) already = blockscale; alpha=1/(a_gs*b_gs).
# For weight, b_blockscale = w_scale (e4m3), and b_gs = 1/w_scale_2 so 1/b_gs = w_scale_2.
b_gs=1.0/w_scale_2.float()
alpha=(1.0/(a_gs*b_gs)).reshape(1).float()

b_sf_mma=swizzle_to_mma(w_scale,N,K)
m_indptr=torch.tensor([0,M],dtype=torch.int32,device="cuda")
out=group_gemm_nvfp4_nt_groupwise(a_q,w_packed.unsqueeze(0),a_sf,b_sf_mma,m_indptr,
    alpha=alpha,out_dtype=torch.bfloat16)
ref=a.float()@w_bf16.float().t()
print("GEMM out",tuple(out.shape),"rel_rms vs dequant-bf16 ref =",round(rel_rms(out,ref),4))
