'''
    model_verification.py

    Script that compare the FLOPs calculated by the cost model (flops.py)
    against the fvcore tool. The comparation is made using the data of
    the GPT2 model and variating S = [32, 128, 512, 1024]

'''


import torch
from transformers import GPT2LMHeadModel
from fvcore.nn import FlopCountAnalysis
from miniroofline.cost_model.flops import flops_prefill_for_model

def verify_model(S:int) -> None:
    torch.set_num_threads(10)
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    model.config.use_cache = False

    input_ids = torch.randint(0, 50000, (1, S))

    with torch.inference_mode():
        fa = FlopCountAnalysis(model, input_ids)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)

    # fvcore side — grouped by module
    print(f"=== fvcore (in FLOPs, ×2 from MACs) for S:{S}===")
    by_mod = fa.by_module()
    total_macs = fa.total()
    fvcore_total_flops = 2 * total_macs # factor of 2 for changing MAC to FLOPs
    print(f"  Grand total:  {fvcore_total_flops/1e9:.3f} G FLOPs   ({total_macs/1e9:.3f} G MACs)")
    print()

    # Aggregation by leaf module names
    attention_macs = mlp_macs = lm_head_macs = ln_macs = 0
    by_op_per_mod = fa.by_module_and_operator()
    for mod_name, ops in by_op_per_mod.items():
        if not ops:
            continue
        total_for_mod = sum(ops.values())
        if total_for_mod == 0:
            continue
        if "attn.c_" in mod_name:   # c_attn or c_proj inside attn
            attention_macs += total_for_mod
        elif "mlp.c_" in mod_name:  # c_fc or c_proj inside mlp
            mlp_macs += total_for_mod
        elif mod_name == "lm_head":
            lm_head_macs += total_for_mod
        elif ".ln_" in mod_name or mod_name.endswith("ln_f"):
            ln_macs += total_for_mod

    print(f'  Attention (c_attn + c_proj):  {2*attention_macs/1e9:.3f} G FLOPs')
    print(f'  MLP       (c_fc + c_proj):    {2*mlp_macs/1e9:.3f} G FLOPs')
    print(f'  LM head:                      {2*lm_head_macs/1e9:.3f} G FLOPs')
    print(f'  LayerNorm:                    {2*ln_macs/1e9:.3f} G FLOPs')
    print(f'  Sum of above:                 {2*(attention_macs+mlp_macs+lm_head_macs+ln_macs)/1e9:.3f} G FLOPs')  

    print()
    print('=== Prediction ===')
    pred = flops_prefill_for_model('gpt2', B=1, S=S)
    for k, v in pred.items():
        if "flops" in k:
            print(f"  {k:<28} {v/1e9:.3f} G FLOPs")

    attention_ratio = pred['attention_proj_flops']/(2*attention_macs)

    mpl_ratio = pred['mlp_flops']/(2*mlp_macs)
    lm_head_ratio = pred['lm_head_flops']/(2*lm_head_macs)
    layer_norm_ratio = pred['layernorm_flops']/(2*ln_macs)
    total_ratio = pred['total_flops']/fvcore_total_flops
    
    return {'attention_ratio': attention_ratio,
             'mpl_ratio':mpl_ratio,
              'lm_head_ratio':lm_head_ratio,
              'layer_norm_ratio':layer_norm_ratio,
              'total_ratio':total_ratio
            }


results = {}
for S in [32, 128, 512, 1024]:
    results[f'{S}'] = verify_model(S)
print()
print('=== Ratios ===')
print('  S\tatte\tmpl\tlm_head\tl_norm\ttotal')
for S, ratios in results.items():
    print(f'  {S}', end='\t')
    for _ , ratio in ratios.items():
        print(f'{ratio:.2f}', end='\t')
    print()    
