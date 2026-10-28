# Start here: train your first generative model

Follow this lab in order. You will check the code, train a flow model from random initialization, generate a labelled digit grid, and then compare diffusion and reflow. Each step tells you what to run, what you should see, and what to read next.

[Browse all code on GitHub](https://github.com/clguo/clguo.github.io/tree/main/tutorials/video-generation/foundations) · [Download code and materials](mnist-generative-models-unified.zip) · [Open the student notebook in Colab](https://colab.research.google.com/github/clguo/clguo.github.io/blob/main/tutorials/video-generation/foundations/student-lab.ipynb) · [Read the theory](index.html)

The notebook mirrors the first five steps below and provides optional cells for the full comparison. Select a GPU runtime before training. Run cells one at a time, read their outputs, and inspect the saved sample grid before moving on.

## 1. Get the code and find your working directory

Choose one route. The ZIP is the smallest download; Git is useful if you want to track your changes.

**ZIP route:** [Download the materials](mnist-generative-models-unified.zip), extract them to a new folder named `mnist-tutorial`, and open a terminal in that folder. The files `requirements.txt`, `digit_model.py`, and the directory `unified/` should be directly inside it.

**Git route:**

```bash
git clone https://github.com/clguo/clguo.github.io.git
cd clguo.github.io/tutorials/video-generation/foundations
```

Run every remaining command from this directory, including commands beginning with `python -m unified`. The relative paths in the training code assume this working directory.

**Checkpoint:** `python -c "from pathlib import Path; print(Path('unified/train.py').is_file())"` prints `True`. If it prints `False`, change into the extracted code folder before installing or running anything.

## 2. Create an environment and check your GPU

Use Python 3.10 or later. On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, use `py -m venv .venv` followed by `.venv\Scripts\Activate.ps1`. A hosted notebook already has an environment; skip virtual-environment creation there.

Install PyTorch using the command from the [official PyTorch installer](https://pytorch.org/get-started/locally/): select your operating system, Pip, Python, and a CUDA build compatible with your NVIDIA GPU driver. Then install this lab's dependencies:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Checkpoint:** `CUDA available: True` is required for the supplied training and full evaluation scripts. A CUDA-enabled PyTorch installation and an allocated NVIDIA GPU are both needed. On a cluster, enter a GPU allocation before running training.

**Without a CUDA GPU:** you can read the code and run the tests on CPU. You can also sample an existing checkpoint with `--device cpu`, more slowly. For this lab's training steps, use a CUDA GPU machine or a GPU notebook runtime. Apple MPS training is not implemented in this trainer.

The download contains source code and published evidence. It does **not** contain trained weights. Following the training steps creates your own checkpoints.

## 3. Verify the mathematical building blocks

```bash
python -m unittest unified.test_unified -v
```

**Checkpoint:** four tests end in `OK`. They check the flow time convention, oracle diffusion sampling, a reference assignment counterexample, and the conditional network's shape and gradients. These tests run without downloading MNIST or training a generator.

Read these files in order, keeping the [full tutorial](index.html) beside the code:

| File | What to understand | Article section |
|---|---|---|
| [`digit_model.py`](https://github.com/clguo/clguo.github.io/blob/main/tutorials/video-generation/foundations/digit_model.py) | Inputs `x`, `t`, `y`; time and class embeddings; the output has the same shape as the image | [What the model sees](index.html#2-what-the-model-sees) |
| [`unified/processes.py`](https://github.com/clguo/clguo.github.io/blob/main/tutorials/video-generation/foundations/unified/processes.py) | How `training_pair` builds a noisy input and a regression target | [Flow matching](index.html#6-flow-first-step-learn-where-pixels-should-move) |
| [`unified/train.py`](https://github.com/clguo/clguo.github.io/blob/main/tutorials/video-generation/foundations/unified/train.py) | Batch → input and target → prediction → loss → backward → optimizer → EMA | [Training loop](index.html#3-the-common-training-loop-line-by-line) |
| [`unified/sample.py`](https://github.com/clguo/clguo.github.io/blob/main/tutorials/video-generation/foundations/unified/sample.py) | Loading EMA weights, fixed noise and labels, calling a sampler, and saving every sample | [Watch generation](index.html#8-watch-the-actual-generation-grouped-by-family) |

**Try explaining it:** for flow training, why is the target `data - noise`, and why is one training update possible without solving an ODE?

## 4. Run a short end-to-end experiment

First verify the whole pipeline with 200 updates. Keep this practice run in its own output directory:

```bash
python -m unified.train --stage flow --steps 200 --batch-size 32 --root results/smoke
python -m unified.sample --checkpoint results/smoke/flow/checkpoint.pt --method fm --nfe 16 --out outputs/smoke-flow.png
```

The trainer downloads MNIST on first use and caches it in `data/`. It finishes by writing a JSON training summary. This short run ends before the normal 1,000-update learning-rate warmup is complete; its purpose is to check data loading, gradients, checkpoint writing, and sampling.

**Checkpoint:** these files now exist:

```text
results/smoke/flow/checkpoint.pt    model, EMA, optimizer, and random states
results/smoke/flow/metadata.json    training configuration and update count
results/smoke/flow/loss.npy         loss history
outputs/smoke-flow.png             your generated sample grid
outputs/smoke-flow.json            sample seed, method, and checkpoint hash
```

Open the PNG. Rows request digits 0–9, with four samples per row. Expect poor or noisy outputs from 200 updates. Getting a valid PNG verifies the pipeline; it does not establish that the model has learned good digits.

**Try changing one thing:** rerun only the sampler with `--seed 2027` and a different `--out` path. The checkpoint stays the same; the starting noise changes.

## 5. Train a full flow model and watch it generate

Start a new run with the published base budget. The default batch size is 128. Use a fresh result directory so the short experiment remains available:

```bash
python -m unified.train --stage flow --steps 20000 --root results/student
python -m unified.sample --checkpoint results/student/flow/checkpoint.pt --method fm --nfe 64 --out outputs/flow-64.png --gif
```

**Checkpoint:** open `outputs/flow-64.png` and `outputs/flow-64.gif`. The GIF records actual intermediate states from this sampling run. Metadata is in `outputs/flow-64.json`. You have trained from random initialization and generated samples without an independent classifier or pretrained generator weights.

The trainer prints progress every 500 updates and saves every 2,000 updates, plus at normal completion. If interrupted, resume from the latest checkpoint with the same planned step count:

```bash
python -m unified.train --stage flow --steps 20000 --root results/student --resume
```

If a run stops before its first saved checkpoint, restart it. Later updates after the most recent checkpoint must be repeated. Runtime depends on hardware and GPU sharing; the article's V100 times describe its recorded experiment.

**Experiment:** keep the checkpoint, seed, and number of samples fixed. Generate at several counted network-call budgets:

```bash
python -m unified.sample --checkpoint results/student/flow/checkpoint.pt --method fm --nfe 1 --out outputs/flow-1.png
python -m unified.sample --checkpoint results/student/flow/checkpoint.pt --method fm --nfe 4 --out outputs/flow-4.png
python -m unified.sample --checkpoint results/student/flow/checkpoint.pt --method fm --nfe 16 --out outputs/flow-16.png
```

Compare them with `flow-64.png`. Inspect variation within each requested digit as well as recognizability. In the published run, one-step base flow produces class prototypes with very little within-class variation.

## 6. Train diffusion once, then compare two samplers

```bash
python -m unified.train --stage diffusion --steps 20000 --root results/student
python -m unified.sample --checkpoint results/student/diffusion/checkpoint.pt --method ddpm --nfe 64 --out outputs/ddpm-64.png
python -m unified.sample --checkpoint results/student/diffusion/checkpoint.pt --method ddim --nfe 64 --out outputs/ddim-64.png
```

**Checkpoint:** both commands use the exact same trained diffusion checkpoint. The fixed seed creates identical initial noise and requested labels. DDPM injects additional noise during sampling; this DDIM configuration is deterministic after initialization.

Read [DDPM training](index.html#4-diffusion-first-step-train-ddpm-to-predict-noise) and [DDIM sampling](index.html#5-diffusion-next-step-ddim-changes-sampling-not-training), then find `eta`'s equivalent in the `sigma` branch of `unified/processes.py`.

**Write down:** which part changed between these two PNGs: the architecture, the training target, the checkpoint, or the sampler?

## 7. Train reflow and its continued-training control

Finish the full base flow from step 5 before generating a teacher bank. The following commands depend on that flow checkpoint:

```bash
python -m unified.train --stage pairs --pairs 20000 --teacher-nfe 128 --root results/student
python -m unified.train --stage reflow --steps 12000 --root results/student
python -m unified.train --stage continued --steps 12000 --root results/student
python -m unified.sample --checkpoint results/student/reflow/checkpoint.pt --method rf2 --nfe 1 --out outputs/reflow-1.png
python -m unified.sample --checkpoint results/student/continued/checkpoint.pt --method continued --nfe 1 --out outputs/continued-1.png
```

**Checkpoint:** the pair stage writes `results/student/pairs/pairs.pt` and its metadata. Reflow and continued training both start from the same base-flow EMA; the former uses fixed teacher pairs, while the latter continues ordinary real-data flow training.

Compare `reflow-1.png`, `continued-1.png`, and `flow-1.png`. Read [what reflow changes](index.html#7-where-rectified-flow-fits-and-what-reflow-changes). Record teacher-pair generation time separately from student training time.

## 8. Reproduce the full measured comparison

Do this after completing diffusion, flow, teacher pairs, reflow, and continued training. Unlike the small sampler above, the full evaluator requires the independent classifier and all five configurations:

```bash
python quality.py --train-classifier
python -m unified.evaluate --root results/student
```

**Checkpoint:** inspect `results/student/evaluation/report.json`. The evaluator keeps 1,000 outputs per configuration and budget, counts network calls, and checks checkpoint and random-stream relationships. Read class agreement, feature distance, and within-class variation together. Their limitations are explained in the article.

The small sampler needs no classifier. Its PNGs are for learning and inspection, while this full evaluator reproduces the numerical protocol. Results can vary with the software stack and hardware.

For the exact article-rendering pipeline, run the complete training stages with the trainer's **default** root `results/unified` instead of `results/student`, then use:

```bash
python -m unified.evaluate
python -m unified.render
```

The renderer currently reads `results/unified/evaluation` and writes `assets/unified_*`. It does not accept a `--root` argument. Do not pass a student result path to that renderer.

## 9. Keep a small experiment notebook

For each change, save the command, checkpoint hash, seed, training updates, sampling calls, a complete sample grid, and your interpretation. Change one variable at a time.

Before moving to DiT, explain the training input, the meaning of the network output, the endpoint distribution, and the sampler update. A useful next exercise is replacing the U-Net backbone while keeping the existing training-pair construction and sampling budget fixed.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `No module named unified` or `digit_model` | Run from the folder containing `digit_model.py` and `unified/`; use `python -m unified.train`, not `python unified/train.py`. |
| `CUDA available: False` | Check that the GPU runtime/allocation is active and that your PyTorch build supports CUDA. Revisit the official installer linked above. |
| `CUDA out of memory` | For a practice run, lower `--batch-size` and choose a fresh `--root`. This changes the protocol, so its result is no longer the published matched-budget experiment. |
| Missing checkpoint | Complete the corresponding training stage. The source ZIP excludes model weights. |
| Poor digits after the smoke run | Expected at 200 updates. Train the separate full run before comparing quality. |
| Reflow cannot find a teacher or pair bank | Finish base flow first, then pairs, then reflow; keep the same `--root` for all dependent stages. |
| Full evaluation cannot find the classifier | Run `python quality.py --train-classifier` first. |
| The first MNIST download fails | Check network access to the dataset host; rerun after removing only the incomplete file named in the error from `data/`. |
