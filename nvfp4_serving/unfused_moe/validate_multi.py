"""Robustness: several seeds/layers/expert-counts, report (1) and (2)."""
import json,sys,torch,flashinfer
sys.path.insert(0,"/work")
from safetensors import safe_open
from unfused_moe import unfused_swigluoai_nvfp4_moe,swigluoai,silu_and_mul,build_expert_weight_scales_mma
ROOT="/models/MiniMax-M3-NVFP4"
idx=json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
_h={}
def get(k):
    f=idx[k];_h.setdefault(f,safe_open(f"{ROOT}/{f}",framework="pt",device="cpu"));return _h[f].get_tensor(k).clone()
def rr(a,b):
    a=a.float();b=b.float();return ((a-b).pow(2).mean().sqrt()/(b.pow(2).mean().sqrt()+1e-20)).item()
_E2M1=torch.tensor([0.,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
def up(p):
    lo=p&0xF;hi=(p>>4)&0xF;o=torch.empty(p.shape[0],p.shape[1]*2,dtype=torch.float32);o[:,0::2]=_E2M1[lo.long()];o[:,1::2]=_E2M1[hi.long()];return o
def dq(w,s,s2):return up(w.cpu())*(s.cpu().float()*s2.cpu().float()).repeat_interleave(16,1)
H=6144;I=3072;dev="cuda"
def load(L,NUM_E):
    w13p=[];w13s=[];w13s2=[];w2p=[];w2s=[];w2s2=[];w13b=[];w2b=[]
    for e in range(NUM_E):
        b=f"model.language_model.layers.{L}.mlp.experts.{e}"
        gw=get(f"{b}.gate_proj.weight");gs=get(f"{b}.gate_proj.weight_scale");g2=get(f"{b}.gate_proj.weight_scale_2")
        uw=get(f"{b}.up_proj.weight");us=get(f"{b}.up_proj.weight_scale");u2=get(f"{b}.up_proj.weight_scale_2")
        dw=get(f"{b}.down_proj.weight");ds=get(f"{b}.down_proj.weight_scale");d2=get(f"{b}.down_proj.weight_scale_2")
        w13p.append(torch.cat([gw,uw],0).cuda());w13s.append(torch.cat([gs,us],0).cuda());w13s2.append(g2.cuda())
        w2p.append(dw.cuda());w2s.append(ds.cuda());w2s2.append(d2.cuda())
        w13b.append(torch.cat([dq(gw,gs,g2),dq(uw,us,u2)],0).cuda().bfloat16());w2b.append(dq(dw,ds,d2).cuda().bfloat16())
    return (torch.stack(w13p),w13s,torch.stack(w13s2).reshape(-1),torch.stack(w2p),w2s,torch.stack(w2s2).reshape(-1),
            torch.stack(w13b),torch.stack(w2b))
def bf16_moe(x,ti,tw,w13b,w2b,act,NUM_E,T):
    out=torch.zeros(T,H,device=dev)
    for e in range(NUM_E):
        sel=(ti==e)
        if not sel.any():continue
        t,sl=sel.nonzero(as_tuple=True)
        gu=x[t].float()@w13b[e].float().t()
        a=swigluoai(gu) if act=="swigluoai" else silu_and_mul(gu)
        out.index_add_(0,t,(a@w2b[e].float().t())*tw[t,sl].float().unsqueeze(-1))
    return out
for (L,NUM_E,T,seed) in [(10,16,96,1234),(20,32,128,7),(5,8,64,99),(40,64,200,3)]:
    w13p,w13s,w13s2,w2p,w2s,w2s2,w13b,w2b=load(L,NUM_E)
    w13m=build_expert_weight_scales_mma(w13s,2*I,H);w2m=build_expert_weight_scales_mma(w2s,H,I)
    torch.manual_seed(seed)
    x=torch.randn(T,H,device=dev,dtype=torch.bfloat16)
    pr=torch.softmax(torch.randn(T,NUM_E,device=dev).float(),-1);tw,ti=torch.topk(pr,4,-1);tw=(tw/tw.sum(-1,keepdim=True))*2.0
    sw=unfused_swigluoai_nvfp4_moe(x,w13p,w13s,w13s2,w2p,w2s,w2s2,ti,tw,activation="swigluoai",w13_sf_mma=w13m,w2_sf_mma=w2m)
    si=unfused_swigluoai_nvfp4_moe(x,w13p,w13s,w13s2,w2p,w2s,w2s2,ti,tw,activation="silu",w13_sf_mma=w13m,w2_sf_mma=w2m)
    rs=bf16_moe(x,ti,tw,w13b,w2b,"swigluoai",NUM_E,T)
    print(f"L={L:2d} E={NUM_E:2d} T={T:3d}: (1)nvfp4-swig vs bf16-swig={rr(sw,rs):.4f}  (2)nvfp4-silu vs bf16-swig={rr(si,rs):.4f}")
