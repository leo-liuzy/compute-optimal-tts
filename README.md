<div align="center">

# Compute-Optimal Test-Time Scaling

[![arXiv](https://img.shields.io/badge/arXiv-2502.06703-ff0000.svg?style=for-the-badge)](https://arxiv.org/abs/2502.06703)  [![Website](https://img.shields.io/badge/Project_Page-000acc?style=for-the-badge&logo=githubpages&logoColor=000&logoColor=white)](https://ryanliu112.github.io/compute-optimal-tts)  [![Github](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/RyanLiu112/compute-optimal-tts)  [![HuggingFace](https://img.shields.io/badge/HugggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/papers/2502.06703)

</div>

<div align="center">
  <p>
    <a href="#🔔news" style="text-decoration: none; font-weight: bold;">🔔 News</a> •
    <a href="#⚙️tts methods" style="text-decoration: none; font-weight: bold;">⚙️ TTS Methods</a> •
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

- [2025/02/11] 🔥 Our paper is released on [arXiv](https://arxiv.org/abs/2502.06703).



## ⚙️ TTS Methods

<img src="./static/images/tts_method.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="tts_method">

## 🏆 Results



<img src="./static/images/small_vs_large.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="small_vs_large">



<img src="./static/images/small_vs_large_FLOPS.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="small_vs_large_FLOPS">



<img src="./static/images/cot_vs_majority_vs_co.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="cot_vs_majority_vs_co">



<img src="./static/images/long-cot.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="long-cot">



## 🚀 Getting Started

We are on a tight budget to clean the code and plan to release the code in **two** days. If you want to try the power of TTS quickly, you can refer to [OpenR](https://github.com/openreasoner/openr), an open-source LLM reasoning repository that we largely refer to.



## 📝 Citation

```
@article{liu2025can,
    title   = {Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling},
    author  = {Runze Liu and Junqi Gao and Jian Zhao and Kaiyan Zhang and Xiu Li and Biqing Qi and Wanli Ouyang and Bowen Zhou},
    journal = {arXiv preprint arXiv:2502.06703},
    year    = {2025}
}
```



## 💡 Acknowledgement

Our code is largely based on [OpenR](https://github.com/openreasoner/openr), an awesome LLM reasoning repository, and their work has been instrumental in our study. We also want to thank the community for providing high-quality open-source PRMs, including [Qwen2.5-Math](https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e), [Skywork-o1](https://huggingface.co/collections/Skywork/skywork-o1-open-67453df58e12f6c3934738d0), [RLHFlow](https://huggingface.co/collections/RLHFlow/rlhflow-math-process-reward-model-6725a42fc8808e12aa1cb144), and [Math-Shepherd](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm).

