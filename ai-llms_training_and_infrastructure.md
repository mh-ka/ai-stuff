# LLMs - Training and infrastructure


## Training, pruning, merging and fine-tuning
- [Fine-tune NLP models](https://towardsdatascience.com/domain-adaption-fine-tune-pre-trained-nlp-models-a06659ca6668)
- [TensorBoard for TensorFlow visualization](https://www.tensorflow.org/tensorboard)
- [How to use consumer hardware to train 70b LLM](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Datasets](https://archive.ics.uci.edu/)
- [RAFT (training strategy for domain specific fine-tuning)](https://gorilla.cs.berkeley.edu/blogs/9_raft.html)
- [How to create large-scale synthetic data for pre-training](https://huggingface.co/blog/cosmopedia)
- [FineWeb Dataset explanation](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [Pints (Model training example)](https://github.com/Pints-AI/1.5-Pints)
- [GenAI project template](https://github.com/AmineDjeghri/generative-ai-project-template)
- [Mergekit (framework for merging LLMs)](https://github.com/arcee-ai/mergekit)
- [Pruning Approach for LLMs (Meta, Bosch)](https://github.com/locuslab/wanda)
- [Abliteration](https://huggingface.co/blog/mlabonne/abliteration)
- [In-browser training playground](https://github.com/trekhleb/homemade-gpt-js)
- [Fine-tuning tutorial for small language models](https://github.com/huggingface/smol-course)
- [Kiln (fine-tuning, synthetic data generation, prompt generation, etc.)](https://github.com/Kiln-AI/Kiln)
- [Train your own reasoning model with unsloth](https://unsloth.ai/blog/r1-reasoning)
- [Fine-tuning OCR model for PDFs](https://github.com/allenai/olmocr)
- [Ultrascale playbook for GPU clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Distillation introduction](https://www.horusailabs.com/blogs/a-primer-to-distillation)
- [Distillation framework](https://github.com/horus-ai-labs/DistillFlow/)
- [LLM Post-Training overview](https://github.com/mbzuai-oryx/Awesome-LLM-Post-training)
- [The role of embeddings](https://www.nomic.ai/blog/posts/embeddings-are-for-so-much-more-than-rag)
- [The state of reinforcement learning](https://sebastianraschka.com/blog/2025/the-state-of-reinforcement-learning-for-llm-reasoning.html)
- [nanoVLM (VLM training)](https://github.com/huggingface/nanoVLM)
- [Agent Reinforcement Trainer](https://github.com/OpenPipe/ART)
- [Atropos (framework for reinforcement learning environment with LLMs)](https://github.com/NousResearch/atropos)
- [ReCall (RL training for Reasoning with tool call)](https://github.com/Agent-RL/ReCall)
- [ms-swift (model fine-tuning framework)](https://github.com/modelscope/ms-swift)
- [Kolo (fine-tuning setup framework)](https://github.com/MaxHastings/Kolo)
- [Reasoning LLM course](https://huggingface.co/reasoning-course)
- [OpenSloth](https://github.com/anhvth/opensloth)
- LLMs from scratch
  - https://github.com/naklecha/llama3-from-scratch
  - https://github.com/meetrais/A-Z-of-Tranformer-Architecture
  - https://github.com/rasbt/LLMs-from-scratch
  - https://stanford-cs336.github.io/spring2025/
- [Adaptive Classifier](https://github.com/codelion/adaptive-classifier)

## Quantization
- [Quantization project](https://github.com/robbiemu/llama-gguf-optimize)
- [Empirical Study of LLaMA3 Quantization](https://arxiv.org/abs/2404.14047)
- [Quantization Evals](https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/)
- [Quantization Types](https://huggingface.co/docs/hub/en/gguf#quantization-types)
- [k-quants explained](https://github.com/ggml-org/llama.cpp/pull/1684)

<blockquote>

The rule is simple:
- FP16 (2 bytes per parameter): VRAM ≈ (B + C × D) × 2
- FP8 (1 byte per parameter): VRAM ≈ B + C × D
- INT4 (0.5 bytes per parameter): VRAM ≈ (B + C × D) / 2

Where B - billions of parameters, C - context size (10M for example), D - model dimensions or hidden_size (e.g. 5120 for Llama 4 Scout).

Some examples for Llama 4 Scout (109B) and full (10M) context window:
- FP8: (109E9 + 10E6 * 5120) / (1024 * 1024 * 1024) ~150 GB VRAM
- INT4: (109E9 + 10E6 * 5120) / 2 / (1024 * 1024 * 1024) ~75 GB VRAM
- 150GB is a single B200 (180GB) (~$8 per hour)
- 75GB is a single H100 (80GB) (~$2.4 per hour)

For 1M context window the Llama 4 Scout requires only 106GB (FP8) or 53GB (INT4 on couple of 5090) of VRAM.

Small quants and 8K context window will give you:
- INT3 (~37.5%) : 38 GB (most of 48 layers are on 5090 GPU)
- INT2 (~25%): 25 GB (almost all 48 layers are on 4090 GPU)
- INT1/Binary (~12.5%): 13 GB (no sure about model capabilities :)

</blockquote>

---

<blockquote>

There’s basically 4 compression techniques that have risen over time: 0, 1, K and I.  They all battle speed, size and accuracy. 0 and 1 were the first, then K, then I. Some platforms have faster implementations of different quant methods as well.  In theory, I is more accurate then K, which is more accurate then 1, which is more accurate then zero, but they will all be close in size.

So on one platform, 0 may be faster than K, but the accuracy is lower.  But on another platform 0 and K will be the same speed, but you want K’s accuracy.

The _M _XL variants take a small but important section of the model and bump it up to 6_K or 8_K, hoping to improve the accuracy for a small size increase.  _XS (extra small) means this was not done. 

And all of the above is theory, you also have to see what happens in reality… it doesn’t always follow the theory.

</blockquote>

---

<blockquote>

Qx means roughly x bits per weight. K_S means the attention weights are S sized (4 bit maybe idrk). K_XL If you ever see it is fp16 or something, L is int8, M is fp6. Generally K_S is fine. Sometimes some combinations perform better, like q5_K_M is worse on benchmarks than q5_K_S on a lot of models even tho it's bigger. q4_K_M and q5_K_S are my go tos.

Q4_K_0 and _1 are older quantization methods I think. I never touch them.

IQ_4_S is a different quantization technique, and it usually has lower perplexity (less deviation from full precision) for the same file size. The XS/S/M/L work the same as Q4_K_M.

Then there's exl quants and awq and what not. EXL quants usually have their bits per weight in the name which makes it easy, and they have lower perplexity for the same size as IQ quants. Have a look at the Exllamav3 repo for a comparison of a few techniques.

K_S model is most recent method, Q4 is decent. 0 and 1 are earlier methods generating the gguf, Only go less than Q4 if you need to compromise over gpu poor and lack of vram. Q4 K_S is a good choice, the Q5 & Q6 barely hold any benefit.

</blockquote>


## Frameworks, stacks, articles etc.

### GPT-2
- [In Rust](https://github.com/felix-andreas/gpt-burn)
- [In llm.c](https://github.com/karpathy/llm.c/discussions/481)
- [In Excel spreadsheet](https://spreadsheets-are-all-you-need.ai/)

### Monitoring
- [Langwatch (LLM monitoring tools suite)](https://github.com/langwatch/langwatch)
- GPU grafana metrics
  - https://github.com/NVIDIA/dcgm-exporter
  - https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/
- [Linux GPU temperature reader](https://github.com/ThomasBaruzier/gddr6-core-junction-vram-temps)
- [Docker GPU power limiting tool](https://github.com/sammcj/NVApi)
- [Wisent-Guard (python monitoring and guardrailing)](https://github.com/wisent-ai/wisent-guard)
- [Management UI for multiple servers](https://github.com/bongobongo2020/nexrift)

### Transformer.js (browser-based AI inference)
- [Github](https://github.com/xenova/transformers.js)
- [v3.2 Release (adds moonshine support)](https://github.com/huggingface/transformers.js/releases/tag/3.2.0)

### Phi-3
- [Phi-3 on device](https://huggingface.co/blog/Emma-N/enjoy-the-power-of-phi-3-with-onnx-runtime)
- [Microsoft's Cookbook on using Phi-3](https://github.com/microsoft/Phi-3CookBook)

### Proxy / Middleware / Mocking
- [Wilmer (LLM Orchestration Middleware)](https://github.com/SomeOddCodeGuy/WilmerAI/)
- [LLM API proxy server](https://github.com/yanolja/ogem)
- [OptiLLM (LLM proxy with optimizations)](https://github.com/codelion/optillm)
- [RouteLLM framework](https://github.com/lm-sys/RouteLLM)
- [Llama Swap (proxy server for model swapping)](https://github.com/mostlygeek/llama-swap)
- [mockllm](https://github.com/stacklok/mockllm)

### Articles and knowledge resources
- [Build private research assistant using llamaindex and llamafile](https://www.llamaindex.ai/blog/using-llamaindex-and-llamafile-to-build-a-local-private-research-assistant)
- [OSS AI Stack](https://www.timescale.com/blog/the-emerging-open-source-ai-stack)
- [Lessons from developing/deploying a copilot](https://www.pulumi.com/blog/copilot-lessons/)
- [List of AI developer tools](https://github.com/sidhq/YC-alum-ai-tools)
- [LLMs and political biases](https://www.technologyreview.com/2023/08/07/1077324/ai-language-models-are-rife-with-political-biases)
- [Reducing costs and improving performance using LLMs](https://portkey.ai/blog/implementing-frugalgpt-smarter-llm-usage-for-lower-costs)
- What We Learned from a Year of Building with LLMs
  - https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i
  - https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii
- [How to get great AI code completions (technical insights on code completion)](https://sourcegraph.com/blog/the-lifecycle-of-a-code-ai-completion)
- [Reverse engineering of Github Copilot extension](https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals.html)
- [AI NPCs in games](https://huggingface.co/blog/npc-gigax-cubzh)
- [Open Source AI Cookbook](https://huggingface.co/learn/cookbook/index)
- [State of AI - China](https://artificialanalysis.ai/downloads/china-report/2025/Artificial-Analysis-State-of-AI-China-Q1-2025.pdf)
- [FastRTC library for building apps with real-time communication](https://huggingface.co/blog/fastrtc)
- [Deepseek's building block optimizations for training / inference infrastructure](https://github.com/deepseek-ai/open-infra-index)
- [Vision Language Models Recap](https://huggingface.co/blog/vlms-2025)
- [How to run QwQ-32B](https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms/qwq-32b-how-to-run-effectively)
- Smolvlm Realtime WebGPU example
  - [Huggingface space](https://huggingface.co/spaces/webml-community/smolvlm-realtime-webgpu/blob/main/index.html)
  - [Github demo](https://github.com/ngxson/smolvlm-realtime-webcam)
- [LLM Model VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)
- [LLM Slop Forensics](https://github.com/sam-paech/slop-forensics)

### Other
- [Docker GenAI Stack](https://github.com/docker/genai-stack)
- [LLM App Stack](https://github.com/a16z-infra/llm-app-stack)
- [Prem AI infrastructure tooling](https://github.com/premAI-io/prem-app)
- [Dify (local LLM app development platform)](https://github.com/langgenius/dify)
- [ExecuTorch (On-Device AI framework)](https://pytorch.org/executorch/stable/intro-overview.html)
- [Code Interpreter SDK (sandbox for LLM code execution)](https://github.com/e2b-dev/code-interpreter)
- [Ratchet (ML developer toolkit for web browser inference)](https://github.com/huggingface/ratchet)
- Meta's Llama Stack
  - [Github](https://github.com/meta-llama/llama-stack)
  - [Apps](https://github.com/meta-llama/llama-stack-apps)
- [PostgresML AI Stack](https://github.com/postgresml/postgresml)
- [Memory Pi (python lib for memory management)](https://github.com/caspianmoon/memoripy)
- [ModernBERT](https://huggingface.co/blog/modernbert)
- [Haystack (LLM app dev framework)](https://github.com/deepset-ai/haystack)
- [Llamaindex (LLM app dev framework)](https://docs.llamaindex.ai/en/stable/#introduction)
- [Using LLM to create a audio storyline](https://github.com/Audio-AGI/WavJourney)
- [LeRobot](https://github.com/huggingface/lerobot)
- [LLM Promptflow](https://microsoft.github.io/promptflow/)
- [prompt-owl (prompt engineering library)](https://github.com/lks-ai/prowl)
- [HuggingFace downloader](https://github.com/huggingface/hf_transfer)
- [Model with symbolic representation (glyphs)](https://github.com/severian42/Computational-Model-for-Symbolic-Representations)
- [Openorch](https://github.com/openorch/openorch)
- [Outlines (structured text generation)](https://github.com/dottxt-ai/outlines)
- [ppllm (cli tool for project-to-prompt)](https://www.npmjs.com/package/ppllm)
- [Dataset questions for testing Chinese models with bias ](https://huggingface.co/datasets/augmxnt/deccp)
- [RecallWeaver (persistent graph memory system)](https://github.com/Asagix/RecallWeaver)
- [lmdeploy (compressing, deploying, serving LLM)](https://github.com/InternLM/lmdeploy)
- [Tensorzero (framework for hosting production-grade LLM application)](https://github.com/tensorzero/tensorzero)
- [Ailoy (lightweight framework for building LLM apps)](https://github.com/brekkylab/ailoy)
- [llm-tools-kiwix (expose offline Kiwix ZIM archives to LLMs)](https://github.com/mozanunal/llm-tools-kiwix)
- [MNN by Alibaba (framework for inference and training)](https://github.com/alibaba/MNN)
- [HF experiment tracking](https://github.com/gradio-app/trackio)
