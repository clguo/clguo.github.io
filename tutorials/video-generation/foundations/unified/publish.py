"""Insert measured evidence into the tutorial and a concise family-grouped comparison."""
import json
from pathlib import Path

NAMES={"ddpm":"DDPM (respaced posterior)","ddim":"DDIM (same checkpoint)","fm":"FM / RF1","rf2":"RF2 (one reflow round)","continued":"FM + 12k ordinary updates"}

def table(report,keys,nfe=64):
    lines=["| Configuration | NFE | Requested-class agreement ↑ | Classifier-feature Fréchet ↓ | Within-class feature variance / real | Seconds / 1,000 images |", "|---|---:|---:|---:|---:|---:|"]
    for key in keys:
        row=report["configurations"][key][str(nfe)]
        lines.append(f"| {NAMES[key]} | {nfe} | {100*row['requested_class_agreement']:.1f}% | {row['classifier_feature_frechet_distance']:.2f} | {row['within_class_feature_variance_ratio']:.3f} | {row['sampler_seconds_including_output_transfer']:.2f} |")
    return "\n".join(lines)


def main():
    path=Path("results/unified/evaluation/report.json");report=json.loads(path.read_text())
    for source,dest in [("blog.md","previous_blog.md"),("index.html","previous_tutorial.html"),("comparison.md","previous_comparison.md"),("comparison.html","previous_comparison.html")]:
        if Path(source).exists() and not Path(dest).exists():Path(dest).write_bytes(Path(source).read_bytes())
    results="### Diffusion family: same denoiser, different sampler\n\n"+table(report,["ddpm","ddim"])
    native=report["configurations"]["ddpm"]["1000"]
    results+=f"\n\nThe separate native 1,000-NFE DDPM reference obtains **{100*native['requested_class_agreement']:.1f}%** requested-class agreement and feature distance **{native['classifier_feature_frechet_distance']:.2f}**, taking **{native['sampler_seconds_including_output_transfer']:.2f} seconds** for 1,000 outputs. The 64-NFE table changes only the sampler; it cannot be interpreted as a comparison between independently trained denoisers.\n\n"
    d64=report['configurations']['ddim']['64']
    results+=(f"At the matched 64-call budget, DDPM and DDIM have very similar runtime because they execute the same network the same number of times. The practical native-chain comparison uses {native['sampler_seconds_including_output_transfer']/d64['sampler_seconds_including_output_transfer']:.1f}× less measured sampling time for DDIM at 64 calls, with the quality tradeoffs shown above. Respaced DDPM also supports this reduced budget. DDIM does not guarantee better quality at equal NFE, and more reverse steps need not monotonically increase a classifier score. One seed cannot establish the significance of small differences.\n\n")
    results+="### Flow family: change the coupling, with a continuation control\n\n"+table(report,["fm","continued","rf2"])
    results+="\n\nThe equal-follow-up-budget comparison is continued FM versus RF2. Both start from the same base EMA and receive 12,000 updates. At low NFE, also inspect whether increased class agreement comes with preserved variation:\n\n"
    results+="| NFE | FM agreement / variance ratio | Continued FM agreement / variance ratio | RF2 agreement / variance ratio |\n|---:|---:|---:|---:|\n"
    for nfe in [1,4,16,64]:
        values=[]
        for key in ["fm","continued","rf2"]:
            row=report["configurations"][key][str(nfe)]
            values.append(f"{100*row['requested_class_agreement']:.1f}% / {row['within_class_feature_variance_ratio']:.3f}")
        results+=f"| {nfe} | "+" | ".join(values)+" |\n"
    f1=report['configurations']['fm']['1'];r1=report['configurations']['rf2']['1']
    f4=report['configurations']['fm']['4'];r4=report['configurations']['rf2']['4'];c4=report['configurations']['continued']['4']
    f64=report['configurations']['fm']['64'];r64=report['configurations']['rf2']['64']
    results+=(f"\n**Read one-step FM carefully:** it scores **{100*f1['requested_class_agreement']:.1f}%** requested-class agreement while its within-class feature variance is **{f1['within_class_feature_variance_ratio']:.3f}×** the real reference and its feature distance is **{f1['classifier_feature_frechet_distance']:.2f}**. In this run it largely produces class prototypes, and ordinary continued training retains this behavior. RF2 reaches **{r1['within_class_feature_variance_ratio']:.3f}×** within-class feature variance and feature distance **{r1['classifier_feature_frechet_distance']:.2f}** in one call, with **{100*r1['requested_class_agreement']:.1f}%** agreement. Read the lower class score alongside its more meaningful visual variation.\n\n"
              f"At four calls, RF2’s feature distance is **{r4['classifier_feature_frechet_distance']:.2f}**, compared with **{f4['classifier_feature_frechet_distance']:.2f}** for base FM and **{c4['classifier_feature_frechet_distance']:.2f}** for equally long continued FM. At 64 calls, the comparison is **{r64['classifier_feature_frechet_distance']:.2f} versus {f64['classifier_feature_frechet_distance']:.2f}** for RF2 and base FM. These measurements support a coarse-integration benefit in this experiment, with extra training and teacher-generation cost; they do not show a high-budget feature-distance improvement.\n")
    results+="\nA higher agreement score is useful only alongside plausible images and variation. Reflow primarily aims to preserve a teacher mapping under coarse integration, not to maximize a classifier score. These measurements describe one training seed; they do not establish a universal ordering of the methods.\n\n"
    real=report['real_vs_real_feature_distance']
    results+=f"The finite-sample real-versus-real feature-distance reference is **{real:.2f}**. On the 1,000-image real reference, requested-class agreement is **{100*report['real_reference_class_agreement']:.1f}%**.\n\n"
    results+="### Training and teacher costs\n\n| Stage | Updates | Seconds | What the cost includes |\n|---|---:|---:|---|\n"
    for key in ["diffusion","flow","reflow","continued"]:
        m=report["checkpoints"][key]["metadata"]
        results+=f"| {key} | {m['steps']:,} | {m['training_seconds']:.1f} | Optimizer updates, EMA and periodic saves |\n"
    pairs=report["pair_generation"]
    results+=f"| Teacher pair generation | — | {pairs['seconds']:.1f} | {pairs['pairs']:,} pairs × {pairs['teacher_nfe']} teacher NFE |\n\n"
    results+="Reflow’s total cost includes base-flow training, teacher-bank generation and student fitting. Continued FM includes base-flow training and its extra real-data updates. DDIM has no additional training cost beyond the diffusion checkpoint. The 1,000-sample evaluation cost is separate from these training costs.\n\n"
    results+="For the three flow checkpoints, the measured dense-path constant-velocity defects (lower means closer to constant velocity along the integrated path, not a quality score) are "
    results+=", ".join(f"**{NAMES[key]}: {report['configurations'][key]['dense_path']['constant_velocity_defect']:.4f}**" for key in ["fm","continued","rf2"])+". They use the same 100 visual inputs and 128 Euler calls.\n"
    template=Path("unified/tutorial.md").read_text()
    assert template.count("@@RESULTS@@")==1
    Path("blog.md").write_text(template.replace("@@RESULTS@@",results))
    comparison="""# Four MNIST Configurations, Organized into Two Families

This is the compact companion to the [full tutorial](index.html) and [pinned repository audit](repository_audit.html). All main panels use one conditional U-Net architecture, identical starting noise and 64 counted network calls. The 100 displayed images per panel are fixed in advance and belong to the full 1,000-output metric set.

![Diffusion together, flow together](assets/unified_samples.png)

## Diffusion: DDPM → DDIM

Train one cosine-schedule noise predictor for 20,000 updates. Keep its checkpoint fixed. Change posterior sampling with fresh noise to deterministic DDIM. Both samplers share conditioning, clipping and the selected time grid. At 64 calls, DDPM explicitly uses a respaced posterior; the native 1,000-call chain is shown separately.

![Diffusion budget comparison](assets/unified_diffusion_budgets.png)

"""+table(report,["ddpm","ddim"])+"""

![Native DDPM reference and reduced-call DDIM](assets/unified_native_diffusion.png)

## Flow: independent FM / RF1 → reflow / RF2

Independent linear-path flow matching and first-stage rectified flow share the objective studied here. Reflow creates paired teacher endpoints, then trains a new velocity field. The student receives 12,000 extra updates and also pays for 20,000 teacher pairs at 128 calls each. Ordinary FM continued for the same 12,000 updates from the same initialization is an additional control.

![Flow budget comparison](assets/unified_flow_budgets.png)

"""+table(report,["fm","continued","rf2"])+"""

![Equally long follow-up training control](assets/unified_reflow_control.png)

## Actual trajectories and diagnostic curves

![Actual four-way generation](assets/unified_generation.gif)

Progress means completed network calls, not identical noise level across families. DDPM is stochastic after initialization; DDIM and Euler flows are deterministic given noise and labels.

![Diagnostics for both families](assets/unified_metrics.png)

Requested-class agreement uses a separate 98.89%-test-accuracy MNIST classifier. It is not human-rated sample validity. Feature Fréchet distance uses that classifier’s 128D features, not Inception. Within-class variation helps expose blurred prototypes but does not by itself establish realistic diversity. No sample is filtered. Shared-V100 timing, one training seed and finite evaluation size limit the conclusions.

The full tutorial explains the equations, code, controls, measured training costs and reproduction commands. The [machine-readable report](results/unified/evaluation/report.json) records every budget, checkpoint hash, per-class agreement and actual call count.
"""
    Path("comparison.md").write_text(comparison)
    print("Published measured tables into blog.md and comparison.md")

if __name__=="__main__":main()
