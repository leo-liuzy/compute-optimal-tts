<div align="center">

# Compute-Optimal Test-Time Scaling

[![arXiv](https://img.shields.io/badge/arXiv-2502.06703-ff0000.svg?style=for-the-badge)](https://arxiv.org/abs/2502.06703)  [![Website](https://img.shields.io/badge/Project_Page-000acc?style=for-the-badge&logo=githubpages&logoColor=000&logoColor=white)](https://ryanliu112.github.io/compute-optimal-tts)  [![Github](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/RyanLiu112/compute-optimal-tts)  [![HuggingFace](https://img.shields.io/badge/HugggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/papers/2502.06703)

</div>

<div align="center">
  <p>
    <a href="#🔔news" style="text-decoration: none; font-weight: bold;">🔔 News</a> •
    <a href="#👀tts methods" style="text-decoration: none; font-weight: bold;">👀 TTS Methods</a> •
    <a href="#🏆results" style="text-decoration: none; font-weight: bold;">🏆 Results</a>
  </p>
  <p>
    <a href="#🚀getting started" style="text-decoration: none; font-weight: bold;">🚀 Getting Started</a> •
    <a href="#📝citation" style="text-decoration: none; font-weight: bold;">📝 Citation</a> •
    <a href="#💡acknowledgement" style="text-decoration: none; font-weight: bold;">💡 Acknowledgement</a>
  </p>
</div>

<img src="./static/images/MATH_co_abs.png" alt="" style="max-width: 100%; height: auto;" id="MATH_co_abs">

## 🔔 News

- **[2025-02-14]** ✨ Code is now available.
- **[2025-02-12]** 📢 Our work is reported by both [QbitAI (量子位)](https://mp.weixin.qq.com/s/BUBp2TShir9MRd6iVtFSfw) and [AI Era (新智元)](https://mp.weixin.qq.com/s/ygv_CIcVJcRsgr98fdKc_g).
- **[2025-02-12]** 🏅 Our paper ranked *#1* on [HuggingFace Daily Papers](https://huggingface.co/papers?date=2025-02-11).
- **[2025-02-11]** 📄 Our paper is released on [arXiv](https://arxiv.org/abs/2502.06703).

## 👀 TTS Methods

<img src="./static/images/tts_method.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="tts_method">

## 🏆 Results



<img src="./static/images/small_vs_large.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="small_vs_large">



<img src="./static/images/small_vs_large_FLOPS.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="small_vs_large_FLOPS">



<img src="./static/images/cot_vs_majority_vs_co.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="cot_vs_majority_vs_co">



<img src="./static/images/long-cot.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="long-cot">



## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/RyanLiu112/compute-optimal-tts.git
cd compute-optimal-tts/src
```

Create a new conda environment and install the dependencies:

```bash
conda create -n tts python=3.10
conda activate tts
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install "ray[default]==2.38.0"
pip install "fschat[model_worker,webui]"
pip install sympy==1.12
cd envs/MATH/latex2sympy
pip install -e .
```

### Supported Models

#### Policy Models

Llama series (Instruct):

- [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f): [8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf): [1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

Qwen series (Instruct):

- [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e): [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), [32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), [72B](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)

DeepSeek-R1-Distill series:

- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

#### Process Reward Models

- [Math-Shepherd](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm): [Math-Shepherd-PRM-7B](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm)
- [RLHFlow](https://huggingface.co/collections/RLHFlow/rlhflow-math-process-reward-model-6725a42fc8808e12aa1cb144): [RLHFlow-PRM-Mistral-8B](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Mistral-Data), [RLHFlow-PRM-Deepseek-8B](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data)
- [Skywork](https://huggingface.co/collections/Skywork/skywork-o1-open-67453df58e12f6c3934738d0): [Skywork-PRM-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B), [Skywork-PRM-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B)
- [Qwen2.5-Math](https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e): [Qwen2.5-Math-PRM-7B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B), [Qwen2.5-Math-PRM-72B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B)

### How to run

#### Best-of-N

Step 1: Generate responses.
```bash
cd src
bash scripts/run.sh --method best_of_n --LM path/to/LM --RM dummy --width 1 --num_seq 1 --num_q 256
```

Step 2: Evaluate responses.
```bash
cd src
bash scripts/run.sh --method best_of_n --LM path/to/LM --RM path/to/RM --width 1 --num_seq 1 --num_q 256
```

#### Beam Search

```bash
cd src
bash scripts/run.sh --method beam_search --LM path/to/LM --RM path/to/RM --width 4 --num_seq 1 --num_q 1
```

#### DVTS

```bash
cd src
bash scripts/run.sh --method beam_search --LM path/to/LM --RM path/to/RM --width 4 --num_seq 1 --num_q 64
```



## 📝 Citation

```bibtex
@article{liu2025can,
    title   = {Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling},
    author  = {Runze Liu and Junqi Gao and Jian Zhao and Kaiyan Zhang and Xiu Li and Biqing Qi and Wanli Ouyang and Bowen Zhou},
    journal = {arXiv preprint arXiv:2502.06703},
    year    = {2025}
}
```



## 💡 Acknowledgement

Our code is largely based on [OpenR](https://github.com/openreasoner/openr), an awesome LLM reasoning repository, and their work has been instrumental in our study. We also want to thank the community for providing high-quality open-source PRMs, including [Qwen2.5-Math](https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e), [Skywork-o1](https://huggingface.co/collections/Skywork/skywork-o1-open-67453df58e12f6c3934738d0), [RLHFlow](https://huggingface.co/collections/RLHFlow/rlhflow-math-process-reward-model-6725a42fc8808e12aa1cb144), and [Math-Shepherd](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm).

