"""End-to-end validation of unfused_swigluoai_nvfp4_moe on REAL M3-NVFP4 experts.

reference: dequant the SAME nvfp4 weights to bf16 and run bf16 GEMM + swigluoai
+ down (the correct target). Compares:
  (1) nvfp4 un-fused (swigluoai) vs bf16-ref(swigluoai)  -> expect ~noise floor
  (2) nvfp4 un-fused (silu)      vs bf16-ref(swigluoai)  -> larger (wrong act)
  (3) b12x_fused_moe(silu)       vs bf16-ref(swigluoai)  -> fused path is wrong
  (4) nvfp4 un-fused(swigluoai)  vs nvfp4 un-fused(silu)  -> swigluoai applied?
"""
import json, sys, torch, flashinfer
sys.path.insert(0, "/work")
from safetensors import safe_open
from unfused_moe import (unfused_swigluoai_nvfp4_moe, swigluoai, silu_and_mul,
    build_expert_weight_scales_mma, SWIGLU_ALPHA, SWIGLU_LIMIT)

ROOT="/models/MiniMax-M3-NVFP4"
idx=json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
_h={}
def get(key):
    f=idx[key]; _h.setdefault(f, safe_open(f"{ROOT}/{f}",framework="pt",device="cpu")); return _h[f].get_tensor(key).clone()
def rel_rms(a,b):
    a=a.float();b=b.float();return ((a-b).pow(2).mean().sqrt()/(b.pow(2).mean().sqrt()+1e-20)).item()
def abs_rms(a,b): return (a.float()-b.float()).pow(2).mean().sqrt().item()
_E2M1=torch.tensor([0.,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
def unpack_e2m1(p):
    lo=p&0x0F;hi=(p>>4)&0x0F;o=torch.empty(p.shape[0],p.shape[1]*2,dtype=torch.float32)
    o[:,0::2]=_E2M1[lo.long()];o[:,1::2]=_E2M1[hi.long()];return o
def dequant(w,s,s2):
    return unpack_e2m1(w.cpu())*(s.cpu().float()*s2.cpu().float()).repeat_interleave(16,dim=1)

L=10; NUM_E=16; H=6144; I=3072; dev="cuda"
w13_packed=[];w13_scale=[];w13_scale_2=[];w2_packed=[];w2_scale=[];w2_scale_2=[];w13_bf16=[];w2_bf16=[]
for e in range(NUM_E):
    b=f"model.language_model.layers.{L}.mlp.experts.{e}"
    gw=get(f"{b}.gate_proj.weight");gs=get(f"{b}.gate_proj.weight_scale");g2=get(f"{b}.gate_proj.weight_scale_2")
    uw=get(f"{b}.up_proj.weight");us=get(f"{b}.up_proj.weight_scale");u2=get(f"{b}.up_proj.weight_scale_2")
    dw=get(f"{b}.down_proj.weight");ds=get(f"{b}.down_proj.weight_scale");d2=get(f"{b}.down_proj.weight_scale_2")
    w13_packed.append(torch.cat([gw,uw],0).cuda());w13_scale.append(torch.cat([gs,us],0).cuda());w13_scale_2.append(g2.cuda())
    w2_packed.append(dw.cuda());w2_scale.append(ds.cuda());w2_scale_2.append(d2.cuda())
    w13_bf16.append(torch.cat([dequant(gw,gs,g2),dequant(uw,us,u2)],0).cuda().bfloat16())
    w2_bf16.append(dequant(dw,ds,d2).cuda().bfloat16())
w13_packed=torch.stack(w13_packed);w2_packed=torch.stack(w2_packed)
w13_scale_2=torch.stack(w13_scale_2).reshape(-1);w2_scale_2=torch.stack(w2_scale_2).reshape(-1)
w13_bf16=torch.stack(w13_bf16);w2_bf16=torch.stack(w2_bf16)
w13_sf_mma=build_expert_weight_scales_mma(w13_scale,2*I,H)
w2_sf_mma=build_expert_weight_scales_mma(w2_scale,H,I)

torch.manual_seed(1234)
T,k=96,4
x=torch.randn(T,H,device=dev,dtype=torch.bfloat16)
router=torch.randn(T,NUM_E,device=dev,dtype=torch.bfloat16)
probs=torch.softmax(router.float(),-1);topk_w,topk_ids=torch.topk(probs,k,-1)
topk_w=(topk_w/topk_w.sum(-1,keepdim=True))*2.0

def bf16_moe(activation):
    out=torch.zeros(T,H,dtype=torch.float32,device=dev)
    for e in range(NUM_E):
        sel=(topk_ids==e)
        if not sel.any():continue
        tok,slot=sel.nonzero(as_tuple=True)
        gate_up=x[tok].float()@w13_bf16[e].float().t()
        act=swigluoai(gate_up,alpha=SWIGLU_ALPHA,limit=SWIGLU_LIMIT) if activation=="swigluoai" else silu_and_mul(gate_up)
        y=act@w2_bf16[e].float().t()
        out.index_add_(0,tok,y*topk_w[tok,slot].float().unsqueeze(-1))
    return out

nvfp4_swig=unfused_swigluoai_nvfp4_moe(x,w13_packed,w13_scale,w13_scale_2,w2_packed,w2_scale,w2_scale_2,
    topk_ids,topk_w,activation="swigluoai",w13_sf_mma=w13_sf_mma,w2_sf_mma=w2_sf_mma)
nvfp4_silu=unfused_swigluoai_nvfp4_moe(x,w13_packed,w13_scale,w13_scale_2,w2_packed,w2_scale,w2_scale_2,
    topk_ids,topk_w,activation="silu",w13_sf_mma=w13_sf_mma,w2_sf_mma=w2_sf_mma)
ref_swig=bf16_moe("swigluoai");ref_silu=bf16_moe("silu")

print("="*74)
print(f"REAL M3-NVFP4 experts layer {L}, E={NUM_E}, T={T}, H={H}, I={I}, top_k={k}")
print("="*74)
print(f"(1) nvfp4-unfused(swigluoai) vs bf16-ref(swigluoai): rel RMS = {rel_rms(nvfp4_swig,ref_swig):.4f}  abs RMS = {abs_rms(nvfp4_swig,ref_swig):.4e}")
print(f"(2) nvfp4-unfused(SILU)      vs bf16-ref(swigluoai): rel RMS = {rel_rms(nvfp4_silu,ref_swig):.4f}")
print(f"(4) nvfp4-unfused(swigluoai) vs nvfp4-unfused(SILU) : rel RMS = {rel_rms(nvfp4_swig,nvfp4_silu):.4f}  [swigluoai truly applied]")
print(f"    bf16-ref(swigluoai)      vs bf16-ref(SILU)      : rel RMS = {rel_rms(ref_swig,ref_silu):.4f}  [activation gap, sanity]")
try:
    from unfused_moe import weight_scale_to_mma
    from flashinfer import b12x_fused_moe
    # b12x needs ONE stacked mma scale tensor (num_groups=E)
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    def stack_mma(lst,N,K):
        Np=((N+127)//128)*128;parts=[]
        for sf in lst:
            sf=sf.view(torch.uint8)
            if sf.shape[0]<Np:sf=torch.cat([sf,torch.zeros(Np-sf.shape[0],sf.shape[1],dtype=torch.uint8,device=dev)],0)
            parts.append(flashinfer.nvfp4_block_scale_interleave(sf))
        return convert_sf_to_mma_layout(torch.cat(parts,0),m=Np,k=K,num_groups=len(lst))
    w1s=stack_mma(w13_scale,2*I,H);w2s=stack_mma(w2_scale,H,I)
    out_b=torch.empty(T,H,dtype=torch.bfloat16,device=dev)
    b12x_fused_moe(x=x,w1_weight=w13_packed,w1_weight_sf=w1s,w1_alpha=w13_scale_2.float(),
        fc2_input_scale=torch.ones(NUM_E,device=dev),w2_weight=w2_packed,w2_weight_sf=w2s,
        w2_alpha=w2_scale_2.float(),token_selected_experts=topk_ids.to(torch.int32),token_final_scales=topk_w,
        num_experts=NUM_E,top_k=k,num_local_experts=NUM_E,output=out_b,activation="silu",quant_mode="nvfp4")
    print(f"(3) b12x_fused_moe(SILU)     vs bf16-ref(swigluoai): rel RMS = {rel_rms(out_b,ref_swig):.4f}")
    print(f"    b12x_fused_moe(SILU)     vs nvfp4-unfused(SILU) : rel RMS = {rel_rms(out_b,nvfp4_silu):.4f}  [fused == our-silu]")
except Exception as ex:
    import traceback;traceback.print_exc();print(f"(3) b12x FAILED: {ex}")
