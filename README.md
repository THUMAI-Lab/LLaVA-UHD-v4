<div align="center">

# LLaVA-UHD v4: What Makes Efficient Visual Encoding in MLLMs?

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.08985) [![Github](https://img.shields.io/badge/LLaVA--UHD%20v4-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/THUMAI-Lab/LLaVA-UHD-v4) [![HF Paper](https://img.shields.io/badge/HF--Paper-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2605.08985)

</div>

<div align="center">
  <p>
    <a href="#-news">🎉 News</a> •
    <a href="#-introduction">📖 Introduction</a> •
    <a href="#-performance">📊 Performance</a> •
    <a href="#-architecture">🏗️ Architecture</a> •
    <a href="#-citation">🎈 Citation</a>
  </p>
</div>

## 🎉 News

- **[Coming Soon]** Evaluation code and model checkpoints will be released before **May 24**.
- **[Coming Soon]** Training code will be released before **June 7**.

## 📖 Introduction

This repository hosts the code and model weights of **LLaVA-UHD v4**, a multimodal large language model (MLLM) designed for efficient high-resolution visual encoding. LLaVA-UHD v4 rethinks the conventional global-encoding-plus-post-ViT-compression paradigm and introduces a **slice-based encoding framework with intra-ViT early compression**. By moving token reduction into shallow ViT layers, our model substantially reduces the computational cost of visual encoding while preserving fine-grained perception ability.

Across eight standard benchmarks covering document understanding, OCR, mathematical reasoning, and general VQA, LLaVA-UHD v4 matches or even surpasses a post-ViT compression baseline under the same 16× final compression ratio, while **reducing visual-encoding FLOPs by 55.8%**. These results demonstrate that aggressive token compression can be performed inside the vision encoder without sacrificing downstream performance, offering a practical path toward scalable high-resolution MLLMs.

## 📊 Performance

<p align="center">
   <img src="./figures/scaling_and_flops.png" alt="Scaling behavior and FLOPs comparison" style="width: 85%;">
</p>

The figure above highlights the core efficiency–performance trade-off of LLaVA-UHD v4. Across training scales from 4M to 64M samples, LLaVA-UHD v4 closely tracks the performance of the strong post-ViT compression baseline, indicating that intra-ViT early compression preserves the model's scaling behavior. At the same time, by moving part of the token reduction into the vision encoder, LLaVA-UHD v4 reduces visual-encoding FLOPs from **3555G to 1573G**, achieving a **55.8% reduction** in computation.

<p align="center">
   <img src="./figures/table.png" alt="Benchmark results" style="width: 85%;">
</p>

## 🏗️ Architecture

<p align="center">
   <img src="./figures/architecture.png" alt="LLaVA-UHD v4 architecture" style="width: 85%;">
</p>

Unlike previous high-resolution MLLMs that encode the full image globally and compress visual tokens only after the ViT, LLaVA-UHD v4 adopts **slice-based encoding** and moves part of the compression directly into the vision encoder. The intra-ViT compressor first performs local window attention to aggregate neighboring visual information, then applies pixel-unshuffle and MLP-based fusion to reduce the token count. As a result, the remaining ViT layers operate on a much shorter visual sequence, substantially lowering the cost of high-resolution visual encoding while maintaining strong fine-grained perception.

## 🧪 Evaluation

### 1) Prepare environment

```bash
cd vlmevalkit
# Use your own virtual environment path
source /path/to/venv/bin/activate
pip install -r requirements.txt
```

If you want `run_eval.sh` to auto-activate your environment, set:

```bash
export VENV_PATH=/path/to/venv
```

Note: some benchmarks require an LLM judge; set `OPENAI_API_KEY` before evaluation.  
If needed, you can also set `OPENAI_API_BASE` (or `OPENAI_API_KEY_JUDGE` / `OPENAI_API_BASE_JUDGE`).

### 2) Run evaluation

```bash
cd vlmevalkit

export MODEL_PATH=/path/to/model_or_checkpoint
export MODEL_NAME=MiniCPM_4_V
export DATASETS="MMMU_DEV_VAL MathVista_MINI MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 MMStar HallusionBench AI2D_TEST OCRBench"
export SAVE_NAME=llava_uhd_v4_eval

# Optional settings
export SAVE_ROOT=/path/to/save/root
export GPU_NUM=8

bash ./scripts/run_eval.sh "$MODEL_PATH" "$MODEL_NAME" "$DATASETS" "$SAVE_NAME"
```

## 🎈 Citation

If you find LLaVA-UHD v4 helpful, please cite us.

```bibtex
@misc{fang2026llavauhdv4makesefficient,
      title={LLaVA-UHD v4: What Makes Efficient Visual Encoding in MLLMs?}, 
      author={Kechen Fang and Yihua Qin and Chongyi Wang and Wenshuo Ma and Tianyu Yu and Yuan Yao},
      year={2026},
      eprint={2605.08985},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.08985}, 
}
```
