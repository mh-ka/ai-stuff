# Language Models

## Basics
- What are LLMs: https://ig.ft.com/generative-ai/
- What are Embeddings: https://simonwillison.net/2023/Oct/23/embeddings/
- Visualization of how a LLM does work: https://bbycroft.net/llm
- How AI works (high level explanation): https://every.to/p/how-ai-works
- How to use AI to do stuff: https://www.oneusefulthing.org/p/how-to-use-ai-to-do-stuff-an-opinionated
- Catching up on the weird world of LLMs: https://simonwillison.net/2023/Aug/3/weird-world-of-llms/
- Prompt Engineering Guide: https://github.com/dair-ai/Prompt-Engineering-Guide
- Tokenization: https://www.youtube.com/watch?v=zduSFxRajkE
- Prompt Injection and Jailbreaking: https://simonwillison.net/2024/Mar/5/prompt-injection-jailbreaking/
- Insights to model file formats: https://vickiboykis.com/2024/02/28/gguf-the-long-way-around/
- Prompt Library: https://www.moreusefulthings.com/prompts
- Mixture of Experts explained (dense vs. sparse models): https://alexandrabarr.beehiiv.com/p/mixture-of-experts
- Cookbook for model creation: https://www.snowflake.com/en/data-cloud/arctic/cookbook/
- Introduction to Vision Language Models: https://arxiv.org/pdf/2405.17247
- Transformer Explainer: https://poloclub.github.io/transformer-explainer/
- A Visual Guide to Quantization: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
- A Visual Guide to Mixture of Experts (MoE): https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts
- Training, Fine-Tuning, Evaluation LLMs: https://www.philschmid.de/
- Explaining how LLMs work: https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels
- Everything I learned so far about running local LLMs: https://nullprogram.com/blog/2024/11/10/
- Uncensored Models: https://erichartford.com/uncensored-models
- 2024 AI Timeline: https://huggingface.co/spaces/reach-vb/2024-ai-timeline
- Latent space explorer: https://www.goodfire.ai/papers/mapping-latent-spaces-llama/
- Interpretability of Neural networks: https://80000hours.org/podcast/episodes/chris-olah-interpretability-research
- GGUF Explanation: https://www.shepbryan.com/blog/what-is-gguf
- Prompting Guide: https://www.promptingguide.ai/
- How I use AI: https://nicholas.carlini.com/writing/2024/how-i-use-ai.html
- My benchmark for LLMs: https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html
- List of relevant European companies in LLM area
  - AI Models: DeepL, Mistral, Silo AI, Aleph Alpha
  - Cloud Hosting: Schwarz Gruppe, Nebius, Impossible Cloud, Delos Cloud, Open Telekom Cloud
- AI and power consumption: https://about.bnef.com/blog/liebreich-generative-ai-the-power-and-the-glory/
- A deep dive into LLMs: https://www.youtube.com/watch?v=7xTGNNLPyMI
- Global AI regulation tracker: https://www.techieray.com/GlobalAIRegulationTracker


## Models

### Major company releases (collections and large models)

#### Meta
- Llama
  - CodeLLama: https://github.com/facebookresearch/codellama
  - slowllama: with offloading to SSD/Memory: https://github.com/okuvshynov/slowllama
  - Llama 3: https://llama.meta.com/llama3/
  - Llama 3.3 70B: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- NLLB (No language left behind): https://github.com/facebookresearch/fairseq/tree/nllb

#### Mistral
- Fine-tune Mistral on your own data: https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb
- Models on huggingface: https://huggingface.co/mistralai (Codestral, Mathstral, Nemo, Mixtral, Mistral Large etc.)
- Uncensored fine-tune: https://huggingface.co/concedo/Beepo-22B
- Mistral Small Instruct: https://huggingface.co/bartowski/Mistral-Small-Instruct-2409-GGUF/blob/main/Mistral-Small-Instruct-2409-Q6_K.gguf

#### DeepSeek
- DeepSeek-LLM 67B: https://github.com/deepseek-ai/DeepSeek-LLM
- DeepSeek-V2-Chat: https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat
- DeepSeek-Coder-V2: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724
- DeepSeek-R1
  - Collection: https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d
  - Model merge: https://huggingface.co/FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview
  - Flash variant (less overthinking)
    - Article: https://novasky-ai.github.io/posts/reduce-overthinking/
    - Model: https://huggingface.co/FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-Flash-32B-Preview
  - Another model merge variant (good for coding): https://huggingface.co/mradermacher/FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview-i1-GGUF

#### Alibaba
- Qwen 1.5: https://qwenlm.github.io/blog/qwen1.5/
- CodeQwen1.5-7B-Chat
  - HuggingFace page: https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat
  - Discussion: https://www.reddit.com/r/LocalLLaMA/comments/1c6ehct/codeqwen15_7b_is_pretty_darn_good_and_supposedly/
- Qwen2: https://qwenlm.github.io/blog/qwen2/
- Qwen2.5
  - Collection: https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e
  - Coder: https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f
  - Uncensored fine-tune: https://huggingface.co/AiCloser/Qwen2.5-32B-AGI
  - Abliterated: https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF
  - Qwentile (custom model merge): https://huggingface.co/maldv/Qwentile2.5-32B-Instruct
  - Reasoning fine-tune: https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview
  - Another variant (cross-architecture distillation of Qwen and Llama): https://huggingface.co/arcee-ai/Virtuoso-Small
  - 1 Million token context window: https://huggingface.co/collections/Qwen/qwen25-1m-679325716327ec07860530ba
- Marco-o1: https://huggingface.co/AIDC-AI/Marco-o1
- Qwen with Questions (Reasoning Preview): https://huggingface.co/bartowski/QwQ-32B-Preview-GGUF
- QwQ-Unofficial-14B-Math-v0.2 (Math and logic fine-tune): https://huggingface.co/bartowski/QwQ-Unofficial-14B-Math-v0.2-GGUF
- QwQ-LCoT-7B-Instruct (reasoning fine-tune): https://huggingface.co/bartowski/QwQ-LCoT-7B-Instruct-GGUF
- QwQ-32B-Preview-IdeaWhiz-v1-GGUF (scientific creativity fine-tune): https://huggingface.co/6cf/QwQ-32B-Preview-IdeaWhiz-v1-GGUF
- Simple reasoning fine-tune example based on Qwen ("test-time scaling"): https://github.com/simplescaling/s1

#### Cohere
- Command-R: https://huggingface.co/CohereForAI/c4ai-command-r-v01
- Command-R 08-2024: https://huggingface.co/bartowski/c4ai-command-r-08-2024-GGUF
- Aya Expanse: https://huggingface.co/collections/CohereForAI/c4ai-aya-expanse-671a83d6b2c07c692beab3c3
- Aya Expanse 32B ungated GGUF: https://huggingface.co/tensorblock/aya-expanse-32b-ungated-GGUF

#### IBM
- Granite Code Models: https://github.com/ibm-granite/granite-code-models
- Granite v3.1: https://huggingface.co/collections/ibm-granite/granite-31-language-models-6751dbbf2f3389bec5c6f02d

#### Other
- Phind-CodeLlama-34B v2: https://huggingface.co/Phind/Phind-CodeLlama-34B-v2
- Prometheus 2 7B/8x7B (LLM for LLM evaluation): https://github.com/prometheus-eval/prometheus-eval
- LLM360's K2 70B (fully open source): https://huggingface.co/LLM360/K2
- Starcoder 2 3B/7B/15B: https://github.com/bigcode-project/starcoder2
- DBRX Base and Instruct MoE 36B: https://github.com/databricks/dbrx
- LagLlama (time series forecasting): https://github.com/time-series-foundation-models/lag-llama
- DiagnosisGPT 6B/34B (medical diagnosis LLM): https://github.com/FreedomIntelligence/Chain-of-Diagnosis
- AI21 Jamba 1.5 12B/94B (hybrid SSM-Transformer): https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini
- Jina Embeddings v3: https://huggingface.co/jinaai/jina-embeddings-v3
- Microsoft's GRIN-MoE: https://huggingface.co/microsoft/GRIN-MoE
- Athene V2 72B (Qwen Fine-Tune): https://huggingface.co/collections/Nexusflow/athene-v2-6735b85e505981a794fb02cc
- LG's EXAONE-3.5 (several sizes): https://huggingface.co/collections/LGAI-EXAONE/exaone-35-674d0e1bb3dcd2ab6f39dbb4
- Nvidia's Llama-3.1-Nemotron-70B: https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
- Gemma 27B (exl2 format): https://huggingface.co/mo137/gemma-2-27b-it-exl2
- Tencent's Hunyuan Large MoE: https://huggingface.co/tencent/Tencent-Hunyuan-Large
- AllenAI's Tulu 3 (Llama3.1 Fine-Tune): https://huggingface.co/collections/allenai/tulu-3-models-673b8e0dc3512e30e7dc54f5

### Smaller model releases (model series and derivatives)

#### Microsoft
- phi-2 2.7B: https://huggingface.co/microsoft/phi-2
- phi-3 / phi-3.5: https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3
- phi-3.5 uncensored: https://huggingface.co/bartowski/Phi-3.5-mini-instruct_Uncensored-GGUF

#### Google
- Gemma 2B and 7B: https://huggingface.co/blog/gemma
- Gemma2 9B: https://huggingface.co/bartowski/gemma-2-9b-it-GGUF
- CodeGemma: https://www.kaggle.com/models/google/codegemma
- RecurrentGemma: https://huggingface.co/google/recurrentgemma-9b
- TigerGemma 9B v3 (fine-tune): https://huggingface.co/TheDrummer/Tiger-Gemma-9B-v3
- SEA-LION (South East Asia fine-tune): https://huggingface.co/aisingapore/gemma2-9b-cpt-sea-lionv3-base
- Gemma with questions (fine-tune): https://huggingface.co/prithivMLmods/GWQ-9B-Preview2

#### Moondream (vision model on edge devices)
- https://github.com/vikhyat/moondream
- https://huggingface.co/vikhyatk/moondream2

#### Mistral (feat. Nvidia)
- NeMo 12B
  - https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
  - https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct
- NeMo Minitron (pruned, distilled, quantized)
  - 4B: https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct
  - 8B: https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base
- Ministral 8B Instruct: https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
- Mistral-Small-24B-Instruct-2501 (aka Mistral Small 3): https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501

#### Apple
- Open-ELM: https://github.com/apple/corenet/tree/main/mlx_examples/open_elm
- Core ML Gallery of on-device models: https://huggingface.co/apple

#### Hugging Face
- SmolLM v1: https://huggingface.co/blog/smollm
- SmolLM v2: https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9

#### Zyphra
- Zamba2-2.7B: https://huggingface.co/Zyphra/Zamba2-2.7B
- Zamba2-1.2B: https://huggingface.co/Zyphra/Zamba2-1.2B
- Zamba2-7B: https://huggingface.co/Zyphra/Zamba2-7B-Instruct

#### Meta
- Llama 3.2 (collection of small models and big vision models): https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf
- MobileLLM: https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95
- Llama3 8B LexiFun Uncensored V1 (fine-tune): https://huggingface.co/Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1
- Nvidia's Nemotron
  - Llama-3.1-Nemotron-70B-Instruct: https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
  - Llama-3.1-Nemotron-51B-Instruct: https://huggingface.co/ymcki/Llama-3_1-Nemotron-51B-Instruct-GGUF

#### WizardLM Uncensored
- 13B: https://huggingface.co/cognitivecomputations/WizardLM-13B-Uncensored
- 7B: https://huggingface.co/cognitivecomputations/WizardLM-7B-Uncensored

#### Cohere
- Aya 23 8B/35B (multilingual specialized): https://huggingface.co/collections/CohereForAI/c4ai-aya-23-664f4cda3fa1a30553b221dc
- Command R7B: https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024

#### Other
- Awesome Small Language Models list: https://github.com/slashml/awesome-small-language-models
- LLMs for on-device deployment: https://github.com/NexaAI/Awesome-LLMs-on-device
- MaLA 7B: https://huggingface.co/MaLA-LM/mala-500
- Yi-9B: https://huggingface.co/01-ai/Yi-9B
- Yi-Coder 9B: https://huggingface.co/01-ai/Yi-Coder-9B-Chat
- AutoCoder 7B: https://github.com/bin123apple/AutoCoder
- Replit Code v1.5 3B for coding: https://huggingface.co/replit/replit-code-v1_5-3b
- RWKV-5 Eagle 7B: https://huggingface.co/RWKV/v5-Eagle-7B-pth
- RefuelLLM-2 8B (data labeling model): https://huggingface.co/refuelai/Llama-3-Refueled
- Doctor Dignity 7B: https://github.com/llSourcell/Doctor-Dignity
- Prem-1B (RAG expert model): https://blog.premai.io/introducing-prem-1b/
- SuperNova-Medius 14B (distillation merge between Qwen and Llama)
  - https://huggingface.co/arcee-ai/SuperNova-Medius
  - https://huggingface.co/bartowski/SuperNova-Medius-GGUF
- CodeGeeX4 9B: https://huggingface.co/THUDM/codegeex4-all-9b
- BigTranslate 13B: https://github.com/ZNLP/BigTranslate
- ContextualAI's OLMoE: https://contextual.ai/olmoe-mixture-of-experts
- OpenCoder 1.5B/8B: https://huggingface.co/collections/infly/opencoder-672cec44bbb86c39910fb55e
- NuExtract 1.5: https://huggingface.co/collections/numind/nuextract-15-670900bc74417005409a8b2d
- Kurage Multilingual 7B: https://huggingface.co/lightblue/kurage-multilingual
- Teuken 7B (based on European languages): https://huggingface.co/openGPT-X/Teuken-7B-instruct-commercial-v0.4
- EuroLLM 9B: https://huggingface.co/utter-project/EuroLLM-9B
- Moxin 7B: https://github.com/moxin-org/Moxin-LLM
- Megrez 3B Instruct: https://huggingface.co/Infinigence/Megrez-3B-Instruct
- GRMR v2.0 (grammar checking): https://huggingface.co/collections/qingy2024/grmr-v20-6759d4172e557af98a2feabc
- Glider (Phi 3.5 Mini based text evaluation model): https://huggingface.co/PatronusAI/glider
- Falcon 3 (several sizes): https://huggingface.co/blog/falcon3
- SmallThinker-3B-Preview: https://huggingface.co/PowerInfer/SmallThinker-3B-Preview
- AMD's OLMo 1B: https://huggingface.co/amd/AMD-OLMo
- Fox-1-1.6B-Instruct: https://huggingface.co/tensoropera/Fox-1-1.6B-Instruct-v0.1
- Pleias models (trained mainly on common corpus): https://huggingface.co/collections/PleIAs/common-models-674cd0667951ab7c4ef84cc4
- AllenAI's OLMo 2: https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc
- InternLM3 8B Instruct: https://huggingface.co/internlm/internlm3-8b-instruct
- Recommended for translation
  - Meta's NLLB: https://huggingface.co/facebook/nllb-200-3.3B
  - ALMA: https://github.com/fe1ixxu/ALMA
  - Unbabel's Towerbase: https://huggingface.co/Unbabel/TowerBase-13B-v0.1
  - Google's MADLAD-400-10B-MT: https://huggingface.co/google/madlad400-10b-mt 
- Selene-1-Mini-Llama-3.1-8B (small evaluation and scoring LLM): https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B
- Arch-Function-3B (function calling): https://huggingface.co/katanemo/Arch-Function-3B
- Confucius-o1-14B (Qwen reasoning fine-tune): https://huggingface.co/netease-youdao/Confucius-o1-14B

### Multimodal / Vision models

#### LLaVA
- Github: https://github.com/haotian-liu/LLaVA
- First Impressions with LLaVA 1.5: https://blog.roboflow.com/first-impressions-with-llava-1-5/
- LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/
- LLaVA-OneVision: https://huggingface.co/collections/lmms-lab/llava-onevision-66a259c3526e15166d6bba37

#### Alibaba
- Qwen-VL: https://qwenlm.github.io/blog/qwen-vl/
- Qwen-VL2
  - https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
  - https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct
- Qwen2-VL: https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d
- Qwen2.5-VL: https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5
- Qwen2-Audio: https://huggingface.co/collections/Qwen/qwen2-audio-66b628d694096020e0c52ff6
- VideoLLaMA3 (video understanding): https://huggingface.co/collections/DAMO-NLP-SG/videollama3-678cdda9281a0e32fe79af15

#### DeepSeek
- DeepSeek-VL: https://github.com/deepseek-ai/DeepSeek-VL
- Janus 1.3B: https://huggingface.co/deepseek-ai/Janus-1.3B
- DeepSeek-VL2 (MoE)
  - Github: https://github.com/deepseek-ai/DeepSeek-VL2
  - HuggingFace: https://huggingface.co/collections/deepseek-ai/deepseek-vl2-675c22accc456d3beb4613ab
- Janus Pro 7B: https://huggingface.co/deepseek-ai/Janus-Pro-7B

#### Google
- PaliGemma: https://www.kaggle.com/models/google/paligemma
- PaliGemma 2: https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48
- Inksight (Hand-writing to text): https://github.com/google-research/inksight/

#### Microsoft
- Florence: https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de
- ComfyUI Florence2 Workflow: https://github.com/kijai/ComfyUI-Florence2
- Phi-3-vision: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cuda
- Florence-VL: https://huggingface.co/jiuhai

#### OpenBMB
- MiniCPM V2.5: https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
- MiniCPM V2.6: https://huggingface.co/openbmb/MiniCPM-V-2_6
- MiniCPM-o 2.6: https://huggingface.co/openbmb/MiniCPM-o-2_6

#### Nvidia
- Eagle: https://huggingface.co/NVEagle/Eagle-X5-13B
- NVLM-D: https://huggingface.co/nvidia/NVLM-D-72B

#### Mistral
- Pixtral 12B: https://huggingface.co/mistralai/Pixtral-12B-2409
- Pixtral 124B: https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411

#### Meta
- Llama 3.2-Vision: https://huggingface.co/collections/meta-llama/
- Apollo 7B (based on Qwen, taken down by Meta): https://huggingface.co/GoodiesHere/Apollo-LMMs-Apollo-7B-t32

#### Other
- Apple's Ferret: https://github.com/apple/ml-ferret
- Idefics2 8B: https://huggingface.co/HuggingFaceM4/idefics2-8b
- Mini-Omni: https://huggingface.co/gpt-omni/mini-omni
- Ultravox: https://github.com/fixie-ai/ultravox
- Molmo: https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19
- Emu3: https://huggingface.co/BAAI/Emu3-Gen
- GOT OCR 2.0: https://github.com/Ucas-HaoranWei/GOT-OCR2.0/
- Ovis 1.6 (vision model with LLM based on Gemma 2 9B)
  - https://github.com/Ucas-HaoranWei/GOT-OCR2.0/
  - https://huggingface.co/stepfun-ai/GOT-OCR2_0
- Aria: https://huggingface.co/rhymes-ai/Aria
- OpenGVLab's InternVL 2.5: https://huggingface.co/collections/OpenGVLab/internvl-25-673e1019b66e2218f68d7c1c
- Megrez 3B Omni: https://huggingface.co/Infinigence/Megrez-3B-Omni
- YuLan Mini: https://github.com/RUC-GSAI/YuLan-Mini
- Bytedance's Sa2VA: https://huggingface.co/collections/ByteDance/sa2va-model-zoo-677e3084d71b5f108d00e093
- LlamaV o1 11B Vision Instruct: https://huggingface.co/omkarthawakar/LlamaV-o1


## Working with LLMs

### Engines for running locally
- vLLM (local python library for running models including vision models): https://docs.vllm.ai/en/stable/index.html
- koboldcpp
  - Default version: https://github.com/LostRuins/koboldcpp
  - AMD ROCM version: https://github.com/YellowRoseCx/koboldcpp-rocm/
- llama.cpp: https://github.com/ggerganov/llama.cpp
- llamafile: https://github.com/Mozilla-Ocho/llamafile

### Tools for running
- Ollama
  - Website: https://ollama.com/
  - Model library: https://ollama.com/library
- LMStudio (desktop app for local model hosting): https://lmstudio.ai/
- Nvidia's ChatRTX (local chat with files and image search): https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/
- Jan app (local model hosting): https://jan.ai/
- Leeroo Orchestrator (choose model depending on task): https://github.com/leeroo-ai/leeroo_orchestrator
- FreedomGPT (local model hosting): https://www.freedomgpt.com/
- ExLlama2 (optimized local inference): https://github.com/turboderp/exllamav2
- pinokio (Simple app for running many AI tools locally): https://pinokio.computer/
- GPT4All (local LLM hosting): https://github.com/nomic-ai/gpt4all
- h2oGPT (Private Q&A with your own documents): https://github.com/h2oai/h2ogpt
- chatd (chat with your own documents): https://github.com/BruceMacD/chatd
- Nextcloud AI assistant: https://github.com/nextcloud/all-in-one
- LocalAI (local server hosting): https://localai.io/
- LLM CLI tool: https://llm.datasette.io/en/stable/
- LibreChat: https://github.com/danny-avila/LibreChat
- Harbor (containerized local LLM hosting)
  - Github: https://github.com/av/harbor
  - Harbor services docs: https://github.com/av/harbor/wiki/2.-Services
- Msty: https://msty.app/
- Any Device: https://github.com/exo-explore/exo
- Local AI Assistant: https://github.com/v2rockets/Loyal-Elephie
- Distributed Llama: https://github.com/b4rtaz/distributed-llama/
- Rust LLM server: https://github.com/gagansingh894/jams-rs
- Ollama, ComfyUI, OpenWebUI on Kubernetes: https://github.com/GoingOffRoading?tab=repositories
- llama.ttf (using font engine to host a LLM): https://fuglede.github.io/llama.ttf
- TransformerLab: https://github.com/transformerlab/transformerlab-app

### Mobile apps
- PocketPal AI: https://github.com/a-ghorbani/pocketpal-ai
- ChatterUI: https://github.com/Vali-98/ChatterUI
- SmolChat Android: https://github.com/shubham0204/SmolChat-Android

### Search engines
- Perplexica: https://github.com/ItzCrazyKns/Perplexica
- OpenPerplex: https://github.com/YassKhazzan/openperplex_backend_os
- Perplexideez: https://github.com/brunostjohn/perplexideez
- Turboseek: https://github.com/Nutlope/turboseek
- MiniSearch: https://github.com/felladrin/MiniSearch
- LLM websearch (very basic): https://github.com/Jay4242/llm-websearch
- Farfalle: https://github.com/rashadphz/farfalle
- More AI Web Search apps: https://www.victornogueira.app/awesome-ai-web-search/
- Phind (AI based search engine): https://www.phind.com/search?home=true
- MiniSearch (LLM based web search): https://github.com/felladrin/MiniSearch

### Model playgrounds
- Poe: https://poe.com/login
- Qolaba: https://www.qolaba.ai/
- ChatLLM: https://chatllm.abacus.ai/

### Coding assistance
- Open Interpreter (local model hosting with code execution and file input): https://github.com/OpenInterpreter/open-interpreter
- Tabby (Self-hosted coding assistant): https://github.com/TabbyML/tabby
- Sweep (AI junior developer): https://github.com/sweepai/sweep
- GPT-engineer (let GPT write and run code): https://github.com/gpt-engineer-org/gpt-engineer
- OSS alternative to Devin AI Software Engineer: https://github.com/stitionai/devika
- OpenDevin: https://github.com/OpenDevin/OpenDevin
- VSCode plugin with Ollama for local coding assistant: https://ollama.com/blog/continue-code-assistant
- Napkins (wireframe to web app conversion): https://github.com/nutlope/napkins
- Continue.dev: https://www.continue.dev/
- Aider (coding): https://github.com/paul-gauthier/aider
- Cursor (GPT-API-based code editor app): https://cursor.sh/
- Plandex (GPT-API-based coding agent): https://github.com/plandex-ai/plandex
- SWE-agent (another GPT-API-based coding agent): https://github.com/princeton-nlp/SWE-agent
- OpenUI (another GPT-API-based coding agent for web components): https://github.com/wandb/openui
- Tabnine: https://www.tabnine.com/
- Melty: https://github.com/meltylabs/melty
- Clone Layout: https://github.com/dmh2000/clone-layout
- tldraw make-real: https://github.com/tldraw/make-real
- Foyle (engineer assistant as VSCode extension): https://foyle.io/docs/overview/
- Codeium: https://codeium.com/
- Cline: https://github.com/cline/cline
- Cline fork with experimental features: https://github.com/RooVetGit/Roo-Cline
- Codai (CLI with RAG): https://github.com/meysamhadeli/codai
- LLM code review: https://github.com/lukasrump/crllm
- Codegate: https://github.com/stacklok/codegate
- Qwen specific coding assistant UI: https://github.com/slyfox1186/script-repo/tree/main/AI/Qwen2.5-Coder-32B-Instruct
- CleanCoderAI: https://github.com/Grigorij-Dudnik/Clean-Coder-AI
- VScode extension (for Fill-in-the-middle, supports Gemini, Deepseek, ...)
  - Github: https://github.com/robertpiosik/gemini-vscode
  - Marketplace: https://marketplace.visualstudio.com/items?itemName=robertpiosik.gemini-coder
- Turn github repo into text ingest for LLM: https://github.com/lcandy2/gitingest-extension
- Source to prompt: https://github.com/Dicklesworthstone/your-source-to-prompt.html
- Agentless: https://github.com/OpenAutoCoder/Agentless
- Source code files as input for LLM 
  - Ingest (source code to markdown as LLM input): https://github.com/sammcj/ingest
  - Repomix: https://github.com/yamadashy/repomix
  - Yek: https://github.com/bodo-run/yek
- What we learned copying all the best code assistants: https://blog.val.town/blog/fast-follow/

### File organizer
- Local File Organizer: https://github.com/QiuYannnn/Local-File-Organizer
- FileWizardAI: https://github.com/AIxHunter/FileWizardAI
- Sortify AI: https://github.com/quentin-r37/sortify-ai

### Deep Research
- From HuggingFace: https://huggingface.co/blog/open-deep-research
- From Firecrawl: https://github.com/nickscamara/open-deep-research
- From Jina AI: https://github.com/jina-ai/node-DeepResearch
- Tutorial from Milvus: https://milvus.io/blog/i-built-a-deep-research-with-open-source-so-can-you.md

### Medical assistant
- Open Health: https://github.com/OpenHealthForAll/open-health
- Health server: https://github.com/seapoe1809/Health_server

### Other
- LLMs comparison for translation: https://www.reddit.com/r/LocalLLaMA/comments/1h4ji6x/comment/lzzsjhi/
- List of models for hosting on 3090: https://www.reddit.com/r/LocalLLaMA/comments/1gai2ol/list_of_models_to_use_on_single_3090_or_4090/
- Scalene (High Performance Python Profiler): https://github.com/plasma-umass/scalene
- Lida (visualization and infographics generated with LLMs): https://github.com/microsoft/lida
- OpenCopilot (LLM integration with provided Swagger APIs): https://github.com/openchatai/OpenCopilot
- LogAI (log analytics): https://github.com/salesforce/logai
- Jupyter AI (jupyter notebook AI assistance): https://github.com/jupyterlab/jupyter-ai
- OCR Tesseract tool (running in browser locally): https://github.com/simonw/tools
- LLM cmd Assistant: https://github.com/simonw/llm-cmd
- Semgrep (autofix using LLMs): https://choly.ca/post/semgrep-autofix-llm/
- GPT Investor: https://github.com/mshumer/gpt-investor
- AI Flow (connect different models): https://ai-flow.net
- Shopping assistant: https://www.claros.so/
- STORM (Wikipedia article creator): https://github.com/stanford-oval/storm
- Open NotebookLM: https://itsfoss.com/open-notebooklm/
- LlavaImageTagger: https://github.com/jabberjabberjabber/LLavaImageTagger
- LLM data analytics: https://github.com/yamalight/litlytics
- Clipboard Conqueror: https://github.com/aseichter2007/ClipboardConqueror
- AnythingLLM: https://github.com/Mintplex-Labs/anything-llm
- Screen Analysis Overlay: https://github.com/PasiKoodaa/Screen-Analysis-Overlay
- Personal Assistant: https://github.com/ErikBjare/gptme/
- Open Canvas: https://github.com/langchain-ai/open-canvas
- GPT-boosted Brainstorming Techniques: https://github.com/Azzedde/brainstormers
- ActuosusAI (chat interaction with word probabilities): https://github.com/TC-Zheng/ActuosusAI
- LynxHub (Local AI Management Hub): https://github.com/KindaBrazy/LynxHub
- Vulnerability Scanner: https://github.com/protectai/vulnhuntr
- Visual environment for prompt engineering: https://www.chainforge.ai
- WikiChat: https://github.com/stanford-oval/WikiChat
- AI Snipping Tool: https://github.com/yannikkellerde/AI-Snip
- Automated AI Web Researcher: https://github.com/TheBlewish/Automated-AI-Web-Researcher-Ollama
- Image Search: https://github.com/0ssamaak0/CLIPPyX
- Vector Companion: https://github.com/SingularityMan/vector_companion
- Mikupad (very simple UI): https://github.com/lmg-anon/mikupad
- Chat with PDF on device: https://github.com/NexaAI/nexa-sdk/tree/main/examples/Chat-with-PDF-locally
- WritingTools: https://github.com/theJayTea/WritingTools
- Local preference/recommendation system for personal data: https://github.com/volotat/Anagnorisis
- AnnotateAI (paper annotation): https://github.com/neuml/annotateai
- Tangent (UI with chats as branches): https://github.com/itsPreto/tangent
- Chat-ext (chrome extension): https://github.com/abhishekkrthakur/chat-ext
- Simple Chat UI: https://github.com/FishiaT/yawullm
- AI Assistant with focus on enterprise environment: https://github.com/onyx-dot-app/onyx
- CLI command explainer: https://github.com/shobrook/wut
- Technical docs creation for python projects: https://github.com/charmandercha/ArchiDoc
- Terminal app prototyping tool: https://github.com/shobrook/termite
- Papeg.ai (chatUI): https://github.com/flatsiedatsie/papeg_ai
- Youtube Summarizer Chrome Extension: https://github.com/avarayr/youtube-summarizer-oss
- Local AI assistant: https://github.com/CNTRLAI/notate
- GraphLLM (graph based data processing with LLMs): https://github.com/matteoserva/GraphLLM
- Context aware translator: https://github.com/CyrusCKF/translator/
- Generative Shell: https://github.com/atinylittleshell/gsh
- Logging assistant creation: https://github.com/jetro30087/Ollogger
- Office plugin for text genAI: https://github.com/suncloudsmoon/TextCraft
- AI dashboard builder (for data analytics): https://github.com/pnmartinez/ai-dashboard-builder
- Local Computer Vision (LCLV) with Moondream: https://github.com/HafizalJohari/lclv
- Website scraper: https://github.com/raznem/parsera
- Bytedance's UI-TARS (automated GUI interaction): https://github.com/bytedance/UI-TARS
- Grammar checking VS code extension: https://marketplace.visualstudio.com/items?itemName=OlePetersen.lm-writing-tool
- LLM calculator (memory estimation): https://github.com/Raskoll2/LLMcalc


## AI developer topics

### Training, quantization, pruning, merging and fine-tuning
- Fine-tune NLP models: https://towardsdatascience.com/domain-adaption-fine-tune-pre-trained-nlp-models-a06659ca6668
- TensorBoard for TensorFlow visualization: https://www.tensorflow.org/tensorboard
- How to use consumer hardware to train 70b LLM: https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Datasets: https://archive.ics.uci.edu/
- RAFT (training strategy for domain specific fine-tuning): https://gorilla.cs.berkeley.edu/blogs/9_raft.html
- Llama 3 from scratch tutorial on GitHub: https://github.com/naklecha/llama3-from-scratch
- How to create large-scale synthetic data for pre-training: https://huggingface.co/blog/cosmopedia
- FineWeb Dataset explanation: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
- Pints (Model training example): https://github.com/Pints-AI/1.5-Pints
- GenAI project template: https://github.com/AmineDjeghri/generative-ai-project-template
- Quantization project: https://github.com/robbiemu/llama-gguf-optimize
- Empirical Study of LLaMA3 Quantization: https://arxiv.org/abs/2404.14047
- Quantization Evals: https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/
- Mergekit (framework for merging LLMs): https://github.com/arcee-ai/mergekit
- Pruning Approach for LLMs (Meta, Bosch): https://github.com/locuslab/wanda
- Abliteration: https://huggingface.co/blog/mlabonne/abliteration
- In-browser training playground: https://github.com/trekhleb/homemade-gpt-js
- Fine-tuning tutorial for small language models: https://github.com/huggingface/smol-course
- Kiln (fine-tuning, synthetic data generation, prompt generation, etc.): https://github.com/Kiln-AI/Kiln
- Train your own reasoning model with unsloth: https://unsloth.ai/blog/r1-reasoning

### Retrieval-Augmented Generation
- RAGs ("Build ChatGPT over your data"): https://github.com/run-llama/rags
- RAGFlow (RAG framework): https://github.com/infiniflow/ragflow
- Mistral Guide on basic RAG: https://docs.mistral.ai/guides/basic-RAG/
- How faithful are RAG models: https://arxiv.org/pdf/2404.10198
- Building search-based RAG: https://simonwillison.net/2024/Jun/21/search-based-rag/
- Langchain documentation on RAG: https://python.langchain.com/v0.2/docs/concepts/#retrieval
- GraphRAG: https://github.com/microsoft/graphrag
- RAG Techniques: https://github.com/NirDiamant/RAG_Techniques
- Controllable RAG Agent: https://github.com/NirDiamant/Controllable-RAG-Agent
- Infinity (AI optimized database): https://github.com/infiniflow/infinity
- DataGemma: https://huggingface.co/collections/google/datagemma-release-66df7636084d2b150a4e6643
- RAGBuilder: https://github.com/kruxai/ragbuilder
- RAG Notes: https://github.com/rmusser01/tldw/blob/main/Docs/RAG_Notes.md
- Too long, did not watch: https://github.com/rmusser01/tldw
- Benchmarking RAG: https://towardsdatascience.com/benchmarking-hallucination-detection-methods-in-rag-6a03c555f063
- Archyve (RAG API): https://github.com/nickthecook/archyve
- Yaraa (RAG Library): https://github.com/khalilbenkhaled/yaraa
- Chainlit RAG (hybrid RAG app): https://github.com/agi-dude/chainlit-rag
- RAG UI: https://github.com/Cinnamon/kotaemon
- RAG database: https://github.com/neuml/txtai
- Search vs. VectorDB: https://blog.elicit.com/search-vs-vector-db/
- Chunking library: https://github.com/bhavnicksm/chonkie
- Open Scholar (research RAG article): https://allenai.org/blog/openscholar
- RAGHub: https://github.com/Andrew-Jang/RAGHub
- Entity-DB (in-browser DB for semantic search): https://github.com/babycommando/entity-db
- Minicheck (fact-checking to reduce hallucinations): https://ollama.com/blog/reduce-hallucinations-with-bespoke-minicheck
- Roaming RAG: https://arcturus-labs.com/blog/2024/11/21/roaming-rag--rag-without-the-vector-database/
- RAG setups for long documents: https://www.reddit.com/r/LocalLLaMA/comments/1hq36dn/practical_online_offline_rag_setups_for_long/
- AutoRAG: https://github.com/Marker-Inc-Korea/AutoRAG
- Volo (local wikipedia RAG): https://github.com/AdyTech99/volo/
- UroBot (medical RAG example): https://github.com/DBO-DKFZ/UroBot
- Legit RAG (RAG Docker setup): https://github.com/Emissary-Tech/legit-rag

### Frameworks, stacks, articles etc.

#### GPT-2
- In Rust: https://github.com/felix-andreas/gpt-burn
- In llm.c: https://github.com/karpathy/llm.c/discussions/481
- In Excel spreadsheet: https://spreadsheets-are-all-you-need.ai/

#### Monitoring
- Langwatch (LLM monitoring tools suite): https://github.com/langwatch/langwatch
- GPU grafana metrics
  - https://github.com/NVIDIA/dcgm-exporter
  - https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/
- Linux GPU temperature reader: https://github.com/ThomasBaruzier/gddr6-core-junction-vram-temps
- Docker GPU power limiting tool: https://github.com/sammcj/NVApi

#### Transformer.js (browser-based AI inference)
- Github: https://github.com/xenova/transformers.js
- v3.2 Release (adds moonshine support): https://github.com/huggingface/transformers.js/releases/tag/3.2.0

#### Phi-3
- Phi-3 on device: https://huggingface.co/blog/Emma-N/enjoy-the-power-of-phi-3-with-onnx-runtime
- Microsoft's Cookbook on using Phi-3: https://github.com/microsoft/Phi-3CookBook

#### Other
- Docker GenAI Stack: https://github.com/docker/genai-stack
- LLM App Stack: https://github.com/a16z-infra/llm-app-stack
- Prem AI infrastructure tooling: https://github.com/premAI-io/prem-app
- Dify (local LLM app development platform): https://github.com/langgenius/dify
- ExecuTorch (On-Device AI framework): https://pytorch.org/executorch/stable/intro-overview.html
- Code Interpreter SDK (sandbox for LLM code execution): https://github.com/e2b-dev/code-interpreter
- Ratchet (ML developer toolkit for web browser inference): https://github.com/huggingface/ratchet
- Build private research assistant using llamaindex and llamafile: https://www.llamaindex.ai/blog/using-llamaindex-and-llamafile-to-build-a-local-private-research-assistant
- Selfie (personalized local text generation): https://github.com/vana-com/selfie
- Meta's Llama Stack
  - Github: https://github.com/meta-llama/llama-stack
  - Apps: https://github.com/meta-llama/llama-stack-apps
- PostgresML AI Stack: https://github.com/postgresml/postgresml
- Wilmer (LLM Orchestration Middleware): https://github.com/SomeOddCodeGuy/WilmerAI/
- LLM API proxy server: https://github.com/yanolja/ogem
- ColiVara (RAG alternative using vision models): https://github.com/tjmlabs/ColiVara
- OptiLLM (LLM proxy with optimizations): https://github.com/codelion/optillm
- Memory management: https://github.com/caspianmoon/memoripy
- Graph-based editor for LLM workflows: https://github.com/PySpur-Dev/PySpur
- OSS AI Stack: https://www.timescale.com/blog/the-emerging-open-source-ai-stack
- CLI tool: https://github.com/cagostino/npcsh
- Microsoft's GenAIScript (LLM orchestrator): https://github.com/microsoft/genaiscript/
- ModernBERT: https://huggingface.co/blog/modernbert
- Haystack (LLM app dev framework): https://github.com/deepset-ai/haystack
- Llamaindex (LLM app dev framework): https://docs.llamaindex.ai/en/stable/#introduction
- Lessons from developing/deploying a copilot: https://www.pulumi.com/blog/copilot-lessons/
- List of AI developer tools: https://github.com/sidhq/YC-alum-ai-tools
- LLMs and political biases: https://www.technologyreview.com/2023/08/07/1077324/ai-language-models-are-rife-with-political-biases
- Using LLM to create a audio storyline: https://github.com/Audio-AGI/WavJourney
- AI for journalism use cases: https://www.youtube.com/watch?v=BJxPKr6ixSM
- Reducing costs and improving performance using LLMs: https://portkey.ai/blog/implementing-frugalgpt-smarter-llm-usage-for-lower-costs
- What We Learned from a Year of Building with LLMs
  - https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i
  - https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii
- How to get great AI code completions (technical insights on code completion): https://sourcegraph.com/blog/the-lifecycle-of-a-code-ai-completion
- AI NPCs in games: https://huggingface.co/blog/npc-gigax-cubzh
- RouteLLM framework: https://github.com/lm-sys/RouteLLM
- Reverse engineering of Github Copilot extension: https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals.html
- LeRobot: https://github.com/huggingface/lerobot
- LLM Promptflow: https://microsoft.github.io/promptflow/
- Vision Models Survey: https://nanonets.com/blog/bridging-images-and-text-a-survey-of-vlms/
- Open Source AI Cookbook: https://huggingface.co/learn/cookbook/index
- LLM Pricing: https://huggingface.co/spaces/philschmid/llm-pricing
- prompt-owl (prompt engineering library): https://github.com/lks-ai/prowl
- PocketFlow (minimalist LLM framework): https://github.com/miniLLMFlow/PocketFlow/
- HuggingFace downloader: https://github.com/huggingface/hf_transfer
- State of AI - China: https://artificialanalysis.ai/downloads/china-report/2025/Artificial-Analysis-State-of-AI-China-Q1-2025.pdf
- Model with symbolic representation (glyphs): https://github.com/severian42/Computational-Model-for-Symbolic-Representations
- Llama Swap (proxy server for model swapping): https://github.com/mostlygeek/llama-swap
- Openorch: https://github.com/openorch/openorch
- Outlines (python package for structured generation): https://dottxt-ai.github.io/outlines/latest/welcome/

### Data extraction
- Pre-processing unstructured data: https://github.com/Unstructured-IO/unstructured
- Crawl4AI (simplify crawling and data extraction as input for LLMs): https://github.com/unclecode/crawl4ai
- URL to LLM input converter: https://github.com/jina-ai/reader
- LLM Scraper (extract structured data from URL): https://github.com/mishushakov/llm-scraper
- Datachain (python library for unstructured data processing): https://github.com/iterative/datachain
- ThePipe (markdown and visuals extraction from PDFs, URLs, etc.): https://github.com/emcf/thepipe
- Firecrawl (websites2markdown): https://github.com/mendableai/firecrawl
- MinerU (PDF data extraction): https://github.com/opendatalab/MinerU
- Nougat (PDF parser): https://github.com/facebookresearch/nougat
- PDF Extract Kit: https://github.com/opendatalab/PDF-Extract-Kit
- Marker (PDF2Markdown): https://github.com/VikParuchuri/marker
- PyMPDF
  - https://github.com/pymupdf/PyMuPDF
  - https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
- Sparrow: https://github.com/katanaml/sparrow
- Extractous: https://github.com/yobix-ai/extractous
- Markitdown (nonAI data extractor provided by Microsoft): https://github.com/microsoft/markitdown
- ExtractThinker: https://github.com/enoch3712/ExtractThinker
- semhash (deduplicating datasets): https://github.com/MinishLab/semhash
- Jina's ReaderLM v2 (HTML to Json): https://huggingface.co/jinaai/ReaderLM-v2
- Docling (document parser): https://github.com/DS4SD/docling
- Surya (OCR): https://github.com/VikParuchuri/surya
- Trafilatura (python library for crawling, scraping, extracting, processing text): https://trafilatura.readthedocs.io/en/latest/
- readability (js library for text extraction): https://github.com/mozilla/readability

### Agents and workflows

#### Microsoft
- OmniParser: https://github.com/microsoft/OmniParser
- TinyTroupe: https://github.com/microsoft/TinyTroupe
- Autogen (agent framework): https://github.com/microsoft/autogen/tree/main
- Autogen magentic one: https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one

#### Coder Agents
- OpenHands (coder agent): https://github.com/All-Hands-AI/OpenHands/
- bolt.diy (coder agent): https://github.com/stackblitz-labs/bolt.diy
- freeact (library for code-based agents): https://github.com/gradion-ai/freeact

#### Other
- How to design an agent for production: https://blog.langchain.dev/how-to-design-an-agent-for-production/
- Document-Oriented Agents: https://towardsdatascience.com/document-oriented-agents-a-journey-with-vector-databases-llms-langchain-fastapi-and-docker-be0efcd229f4
- Experts.js (Multi AI Agent Systems Framework in Javascript): https://github.com/metaskills/experts
- Pipecat (build conversational agents): https://github.com/pipecat-ai/pipecat
- AI Agent Infrastructure: https://www.madrona.com/the-rise-of-ai-agent-infrastructure/
- Qwen Agent framework: https://github.com/QwenLM/Qwen-Agent
- Mem-Zero (memory layer for AI agents): https://github.com/mem0ai/mem0
- OS-Atlas: https://github.com/OS-Copilot/OS-Atlas
- Arch (prompt gateway): https://github.com/katanemo/archgw
- Letta (agents with memory): https://github.com/letta-ai/letta
- Building Effective Agents (article from Anthropic): https://www.anthropic.com/research/building-effective-agents
- OpenAI's Swarm (agent orchestrator): https://github.com/openai/swarm
- Amazon's Multi-Agent orchestrator: https://github.com/awslabs/multi-agent-orchestrator
- inferable (scalable agentic automation): https://github.com/inferablehq/inferable
- Smolagents (python based agent library with focus on simplicity): https://github.com/huggingface/smolagents
- n8n (LLM workflow editor): https://github.com/n8n-io/n8n
- Flowise (LLM workflow editor): https://github.com/FlowiseAI/Flowise
- Omnichain (LLM workflow editor): chttps://github.com/zenoverflow/omnichain
- Agents - tools and planning: https://huyenchip.com/2025/01/07/agents.html
- Agent recipes: https://www.agentrecipes.com/
- phidata (framework for building multi-modal agents): https://github.com/phidatahq/phidata
- langroid (framework to easily build LLM-powered applications): https://github.com/langroid/langroid
- Agent stack example
  - AG2: https://github.com/ag2ai/ag2
  - Guidance for constraint generation: https://github.com/guidance-ai/guidance
  - ChromaDB as a vector store: https://github.com/chroma-core/chroma
  - Stella v5 for embeddings: https://huggingface.co/dunzhang/stella_en_400M_v5
  - Marker for converting PDFs to Markdown: https://github.com/VikParuchuri/marker
  - Markitdown for converting non-PDFs to Markdown: https://github.com/microsoft/markitdown
  - JoyCaption for image captioning: https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava
- Different agent definitions summed up with LLM: https://gist.github.com/simonw/beaa5f90133b30724c5cc1c4008d0654
- Atomic agents: https://github.com/BrainBlend-AI/atomic-agents
- Goose: https://github.com/block/goose
- Manifold (workflow editor): https://github.com/intelligencedev/manifold
- Intellagent (advanced multi-agent framework for evaluation and optimization of conversational agents): https://github.com/plurai-ai/intellagent


## Evaluation and leaderboards

### Links
- How to construct domain-specific LLM evaluation systems: https://hamel.dev/blog/posts/evals/
- Big Code Models Leaderboard: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
- Chatbot Arena: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
- Coding Benchmark: https://prollm.toqan.ai/leaderboard
- CanAiCode Leaderboard: https://huggingface.co/spaces/mike-ravkine/can-ai-code-results
- OpenCompass 2.0: https://github.com/open-compass/opencompass
- Open LLM Progress Tracker: https://huggingface.co/spaces/andrewrreed/closed-vs-open-arena-elo
- MMLU Pro: https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro
- LLM API Performance Leaderboard: https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard
- SEAL Leaderboard: https://scale.com/leaderboard
- Lightweight Library from OpenAI to evaluate LLMs: https://github.com/openai/simple-evals
- Open LLM Leaderboard 2: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- Open VLM Leaderboard: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
- Skeleton Key Jailbreak: https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique
- LLM Eval simplified: https://www.philschmid.de/llm-evaluation
- LiveBench: https://livebench.ai/
- Benchmark Aggregator: https://benchmark-aggregator-lvss.vercel.app/
- LLM Comparison: https://artificialanalysis.ai/
- LLM Planning Benchmark: https://github.com/karthikv792/LLMs-Planning
- LLM Cross Capabilities: https://github.com/facebookresearch/llm-cross-capabilities
- MLE-Bench (Machine Learning tasks benchmark): https://github.com/openai/mle-bench
- Compliance EU AI Act Framework: https://github.com/compl-ai/compl-ai
- Open Financial LLM Leaderboard: https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard
- Aider Leaderboards: https://aider.chat/docs/leaderboards/
- BigCodeBench: https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard
- BIG-bench: https://github.com/google/BIG-bench
- TruthfulQA: https://github.com/sylinrl/TruthfulQA
- FACTS Leaderboard: https://www.kaggle.com/facts-leaderboard
- Massive Text Embedding Benchmark Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Reasoning challenges prompts: https://github.com/cpldcpu/MisguidedAttention
- Evaluation model: https://huggingface.co/Unbabel/XCOMET-XL
- UGI (Uncensored) Leaderboard: https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard
- GPU Poor LLM Arena: https://huggingface.co/spaces/k-mktr/gpu-poor-llm-arena
- Several LLM benchmarks: https://github.com/lechmazur
- Creative writing benchmark: https://eqbench.com/creative_writing.html

### Benchmark List

| Benchmark | Measures | Key Features |
|-----------|----------|--------------|
| 1. GPQA | - Challenging graduate-level questions in science<br>- Requires deep specialized understanding | - Extreme difficulty (~65% expert accuracy)<br>- Domain-specific expertise<br>- Scalable oversight experiments |
| 2. MMLU | - General knowledge across 57 subjects<br>- World knowledge and reasoning | - Broad topic coverage<br>- Zero-shot/few-shot evaluation<br>- Multiple-choice scoring |
| 3. MMLU-Pro | - Enhanced reasoning-focused questions<br>- Increased complexity | - Ten answer choices<br>- More reasoning-intensive<br>- Greater prompt stability |
| 4. MATH | - Mathematical problem-solving | - High school to competition-level problems<br>- Covers multiple math domains<br>- Detailed step-by-step solutions |
| 5. HumanEval | - Functional code generation correctness | - Code generation from docstrings<br>- Pass@k evaluation metric<br>- Simulates real-world coding |
| 6. MMMU | - Multimodal reasoning<br>- College-level subject knowledge | - Text and image understanding<br>- 183+ subfields<br>- Expert-level questions |
| 7. MathVista | - Mathematical reasoning in visual contexts | - Combines math and graphical tasks<br>- Highlights visual reasoning gaps |
| 8. DocVQA | - Question answering from document images | - Document element interpretation<br>- Real-world document analysis<br>- Normalized Levenshtein Similarity |
| 9. HELM | - Comprehensive model performance | - Multifaceted assessment<br>- Error analysis<br>- Diverse task coverage |
| 10. GLUE | - General language understanding | - Multiple NLP tasks<br>- Public datasets<br>- Performance leaderboard |
| 11. BIG-Bench Hard | - Model limitations and challenges | - 23 tasks beyond human-rater performance<br>- Focuses on reasoning boundaries |
| 12. MT-Bench | - Conversational coherence<br>- Instruction-following | - Multi-turn conversations<br>- LLM-as-a-Judge<br>- Human preference annotations |
| 13. FinBen | - Financial domain capabilities | - 36 datasets across 24 financial tasks<br>- Practical financial testing |
| 14. LegalBench | - Legal reasoning capabilities | - Collaborative legal task development<br>- Real-world legal scenario simulation |

Source: https://www.reddit.com/r/LocalLLaMA/comments/1i4l5hb/what_llm_benchmarks_actually_measure_explained/
