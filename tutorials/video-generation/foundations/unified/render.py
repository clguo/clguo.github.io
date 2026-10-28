"""Render unfiltered, fixed-index evidence from unified.evaluate; no generation here."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT=Path("results/unified/evaluation")
OUT=Path("assets");OUT.mkdir(exist_ok=True)
NAMES={"ddpm":"DDPM · stochastic posterior", "ddim":"DDIM · deterministic", "fm":"Flow matching / RF1", "rf2":"Rectified flow / RF2", "continued":"FM + 12k ordinary updates"}
COLORS={"ddpm":"#7952b3","ddim":"#b45a96","fm":"#147f93","rf2":"#cb7416","continued":"#687386"}
from matplotlib import font_manager
FONT_PATH=font_manager.findfont("DejaVu Sans")
def font(size):
    try:return ImageFont.truetype(FONT_PATH,size)
    except OSError:return ImageFont.load_default()

def tile(samples,columns=10,scale=2):
    pixels=((samples.detach().float().cpu().clamp(-1,1)+1)*127.5).byte().numpy()[:,0]
    rows=(len(pixels)+columns-1)//columns
    canvas=Image.new("RGB",(columns*29-1,rows*29-1),(15,20,29))
    for i,pixel in enumerate(pixels):canvas.paste(Image.fromarray(pixel).convert("RGB"),((i%columns)*29,(i//columns)*29))
    return canvas.resize((canvas.width*scale,canvas.height*scale),Image.Resampling.NEAREST)

def four_panel(values,step=64,total=64):
    canvas=Image.new("RGB",(1240,1430),"#f7f9fc");d=ImageDraw.Draw(canvas)
    d.text((28,14),"Two families, four sampling configurations",font=font(29),fill="#17253c")
    d.text((28,55),f"{step}/{total} network calls · identical initial noise and requested labels",font=font(19),fill="#526078")
    for row,keys in enumerate([("ddpm","ddim"),("fm","rf2")]):
        y=92+row*663
        d.text((28,y),"DIFFUSION: DDPM → DDIM" if row==0 else "FLOW: independent FM / RF1 → reflow / RF2",font=font(23),fill="#305db7")
        for col,key in enumerate(keys):
            x=36+col*610
            d.text((x,y+34),NAMES[key],font=font(21),fill="#17253c")
            canvas.paste(tile(values[key]),(x,y+68))
    d.text((28,1410),"Rows request digits 0–9. First 10 per class, fixed before evaluation. No filtering.",font=font(15),fill="#526078")
    return canvas

def budget_panel(endpoints,keys,filename):
    canvas=Image.new("RGB",(1280,80+len(keys)*250),"#f7f9fc");d=ImageDraw.Draw(canvas)
    d.text((20,12),"Same noise and labels across budgets · unfiltered samples",font=font(24),fill="#17253c")
    indices=torch.cat([torch.arange(2)+100*k for k in range(10)])
    for col,nfe in enumerate([1,4,16,64]):
        d.text((245+col*255,49),f"{nfe} NFE",font=font(21),fill="#305db7")
        for row,key in enumerate(keys):
            if col==0:d.text((15,100+row*250),{"ddpm":"DDPM\nstochastic posterior","ddim":"DDIM\ndeterministic","fm":"FM / RF1","rf2":"RF2\none reflow round","continued":"FM + 12k\nordinary updates"}[key],font=font(18),fill="#17253c")
            canvas.paste(tile(endpoints[key][str(nfe)][indices],columns=5,scale=1).resize((244,195),Image.Resampling.NEAREST),(245+col*255,80+row*250))
    canvas.save(OUT/filename)

def main():
    torch.set_num_threads(2)
    report=json.loads((ROOT/"report.json").read_text())
    keys=["ddpm","ddim","fm","rf2","continued"]
    endpoints={key:torch.load(ROOT/f"{key}_endpoints.pt",weights_only=True) for key in keys}
    paths={key:torch.load(ROOT/f"{key}_trajectory.pt",weights_only=True) for key in keys}
    visual=torch.cat([torch.arange(10)+100*k for k in range(10)])
    four_panel({key:endpoints[key]["64"][visual] for key in keys}).save(OUT/"unified_samples.png")
    steps=list(range(0,65,2))
    frames=[four_panel({key:paths[key][i] for key in keys},i) for i in steps]
    frames[0].save(OUT/"unified_generation.gif",save_all=True,append_images=frames[1:],duration=[150]*32+[2000],loop=0)
    budget_panel(endpoints,["ddpm","ddim"],"unified_diffusion_budgets.png")
    budget_panel(endpoints,["fm","rf2"],"unified_flow_budgets.png")
    budget_panel(endpoints,["fm","continued","rf2"],"unified_reflow_control.png")
    # A native-chain reference is labelled separately from the matched-64-NFE comparison.
    canvas=Image.new("RGB",(1240,690),"#f7f9fc");d=ImageDraw.Draw(canvas)
    for col,(key,budget) in enumerate([("ddpm",1000),("ddim",64)]):
        x=30+col*610;d.text((x,15),f"{NAMES[key]} · {budget} NFE",font=font(22),fill="#17253c")
        canvas.paste(tile(endpoints[key][str(budget)][visual]),(x,65))
    canvas.save(OUT/"unified_native_diffusion.png")
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout="constrained")
    metrics=[("requested_class_agreement","Requested-class agreement",True),("classifier_feature_frechet_distance","Classifier-feature Fréchet ↓",False),("within_class_feature_variance_ratio","Within-class feature variance / real",False)]
    for row,family in enumerate([["ddpm","ddim"],["fm","continued","rf2"]]):
        for col,(metric,title,percent) in enumerate(metrics):
            ax=axes[row,col]
            for key in family:
                xs=[1,4,16,64];ys=[report["configurations"][key][str(x)][metric] for x in xs]
                ax.plot(xs,np.array(ys)*(100 if percent else 1),"o-",label=NAMES[key],color=COLORS[key])
            if col==2:ax.axhline(1,color="gray",ls=":",label="real reference")
            ax.set(xscale="log",xticks=[1,4,16,64],xticklabels=[1,4,16,64],xlabel="Counted network evaluations per image",title=title)
            if col==1:
                ax.set_yscale("log")
                ax.set_ylabel("Distance (log scale)")
                ax.axhline(report["real_vs_real_feature_distance"],color="gray",ls=":",label="real vs real")
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("1,000 unfiltered conditional samples per point · one training seed",fontsize=14)
    fig.savefig(OUT/"unified_metrics.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for key,stage,axis in [("ddpm","diffusion",0),("fm","flow",1),("continued","continued",1),("rf2","reflow",1)]:
        loss=np.load(Path("results/unified")/stage/"loss.npy")
        size=200;n=len(loss)//size
        axes[axis].plot(np.arange(n)*size+(20000 if stage in ["continued","reflow"] else 0),loss[:n*size].reshape(n,size).mean(1),label=NAMES[key],color=COLORS[key])
    axes[0].set(title="Diffusion: noise-prediction MSE",xlabel="Optimizer updates",ylabel="MSE")
    axes[1].set(title="Flow: velocity-prediction MSE",xlabel="Updates (follow-ups begin at 20k)",ylabel="MSE; objectives / targets change after 20k")
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/"unified_losses.png",dpi=160);plt.close(fig)
    print("Rendered grouped comparisons and actual sampling trajectories")

if __name__=="__main__":main()
