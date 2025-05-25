# LLMs - Basics and Models

## Basics
- [What are LLMs](https://ig.ft.com/generative-ai/)
- [What are Embeddings](https://simonwillison.net/2023/Oct/23/embeddings/)
- [Visualization of how a LLM does work](https://bbycroft.net/llm)
- [How AI works (high level explanation)](https://every.to/p/how-ai-works)
- [How to use AI to do stuff](https://www.oneusefulthing.org/p/how-to-use-ai-to-do-stuff-an-opinionated)
- [Catching up on the weird world of LLMs](https://simonwillison.net/2023/Aug/3/weird-world-of-llms/)
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Prompt Injection and Jailbreaking](https://simonwillison.net/2024/Mar/5/prompt-injection-jailbreaking/)
- [Insights to model file formats](https://vickiboykis.com/2024/02/28/gguf-the-long-way-around/)
- [Prompt Library](https://www.moreusefulthings.com/prompts)
- [Mixture of Experts explained (dense vs. sparse models)](https://alexandrabarr.beehiiv.com/p/mixture-of-experts)
- [Cookbook for model creation](https://www.snowflake.com/en/data-cloud/arctic/cookbook/)
- [Introduction to Vision Language Models](https://arxiv.org/pdf/2405.17247)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
- [Training, Fine-Tuning, Evaluation LLMs](https://www.philschmid.de/)
- [Explaining how LLMs work](https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels)
- [Everything I learned so far about running local LLMs](https://nullprogram.com/blog/2024/11/10/)
- [Uncensored Models](https://erichartford.com/uncensored-models)
- [2024 AI Timeline](https://huggingface.co/spaces/reach-vb/2024-ai-timeline)
- [Latent space explorer](https://www.goodfire.ai/papers/mapping-latent-spaces-llama/)
- [Interpretability of Neural networks](https://80000hours.org/podcast/episodes/chris-olah-interpretability-research)
- [GGUF Explanation](https://www.shepbryan.com/blog/what-is-gguf)
- [Prompting Guide](https://www.promptingguide.ai/)
- [How I use AI](https://nicholas.carlini.com/writing/2024/how-i-use-ai.html)
- [My benchmark for LLMs](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html)
- List of relevant European companies in LLM area
  - AI Models: DeepL, Mistral, Silo AI, Aleph Alpha
  - Cloud Hosting: Schwarz Gruppe, Nebius, Impossible Cloud, Delos Cloud, Open Telekom Cloud
- [AI and power consumption](https://about.bnef.com/blog/liebreich-generative-ai-the-power-and-the-glory/)
- [A deep dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI)
- [Global AI regulation tracker](https://www.techieray.com/GlobalAIRegulationTracker)


## Models

### Major company releases (collections and large models)

#### Meta
- Llama
  - [CodeLLama](https://github.com/facebookresearch/codellama)
  - [slowllama: with offloading to SSD/Memory](https://github.com/okuvshynov/slowllama)
  - [Llama 3](https://llama.meta.com/llama3/)
  - [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [NLLB (No language left behind)](https://github.com/facebookresearch/fairseq/tree/nllb)
- Nvidia's Llama Nemotron
  - [Collection](https://huggingface.co/collections/nvidia/llama-nemotron-67d92346030a2691293f200b)

#### Mistral
- [Fine-tune Mistral on your own data](https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb)
- [Models on huggingface](https://huggingface.co/mistralai (Codestral, Mathstral, Nemo, Mixtral, Mistral Large etc.))
- [Uncensored fine-tune](https://huggingface.co/concedo/Beepo-22B)
- [Mistral Small Instruct](https://huggingface.co/bartowski/Mistral-Small-Instruct-2409-GGUF/blob/main/Mistral-Small-Instruct-2409-Q6_K.gguf)

#### DeepSeek
- [DeepSeek-LLM 67B](https://github.com/deepseek-ai/DeepSeek-LLM)
- [DeepSeek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)
- [DeepSeek-Coder-V2](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724)
- DeepSeek-R1
  - [Collection](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d)
  - [Model merge](https://huggingface.co/FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview)
  - Flash variant (less overthinking)
    - [Article](https://novasky-ai.github.io/posts/reduce-overthinking/)
    - [Model](https://huggingface.co/FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-Flash-32B-Preview)
  - [Another model merge variant (good for coding)](https://huggingface.co/mradermacher/FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview-i1-GGUF)

#### Alibaba
- [Qwen 1.5](https://qwenlm.github.io/blog/qwen1.5/)
- [Qwen2](https://qwenlm.github.io/blog/qwen2/)
- Qwen2.5
  - [Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
  - [Coder](https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f)
  - [Uncensored fine-tune](https://huggingface.co/AiCloser/Qwen2.5-32B-AGI)
  - [Abliterated](https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF)
  - [Qwentile (custom model merge)](https://huggingface.co/maldv/Qwentile2.5-32B-Instruct)
  - [Reasoning fine-tune](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview)
  - [Another variant (cross-architecture distillation of Qwen and Llama)](https://huggingface.co/arcee-ai/Virtuoso-Small)
  - [1 Million token context window](https://huggingface.co/collections/Qwen/qwen25-1m-679325716327ec07860530ba)
- [Marco-o1](https://huggingface.co/AIDC-AI/Marco-o1)
- [Qwen with Questions (Reasoning Preview)](https://huggingface.co/bartowski/QwQ-32B-Preview-GGUF)
- [QwQ-Unofficial-14B-Math-v0.2 (Math and logic fine-tune)](https://huggingface.co/bartowski/QwQ-Unofficial-14B-Math-v0.2-GGUF)
- [QwQ-LCoT-7B-Instruct (reasoning fine-tune)](https://huggingface.co/bartowski/QwQ-LCoT-7B-Instruct-GGUF)
- [QwQ-32B-Preview-IdeaWhiz-v1-GGUF (scientific creativity fine-tune)](https://huggingface.co/6cf/QwQ-32B-Preview-IdeaWhiz-v1-GGUF)
- [Simple reasoning fine-tune example based on Qwen ("test-time scaling")](https://github.com/simplescaling/s1)
- [Qwen QwQ 32B Reasoning Model](https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF)

#### Cohere
- [Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
- [Command-R 08-2024](https://huggingface.co/bartowski/c4ai-command-r-08-2024-GGUF)
- [Command-R 03-2025](https://huggingface.co/CohereForAI/c4ai-command-a-03-2025)
- [Aya Expanse](https://huggingface.co/collections/CohereForAI/c4ai-aya-expanse-671a83d6b2c07c692beab3c3)
- [Aya Expanse 32B ungated GGUF](https://huggingface.co/tensorblock/aya-expanse-32b-ungated-GGUF)

#### Google
- [Gemma-3-27B-IT-QAT-GGUF](https://huggingface.co/ubergarm/gemma-3-27b-it-qat-GGUF)
- [Fallen-Gemma3-27B-v1 (uncensored)](https://huggingface.co/TheDrummer/Fallen-Gemma3-27B-v1)

#### Other
- [Phind-CodeLlama-34B v2](https://huggingface.co/Phind/Phind-CodeLlama-34B-v2)
- [LLM360's K2 70B (fully open source)](https://huggingface.co/LLM360/K2)
- [DBRX Base and Instruct MoE 36B](https://github.com/databricks/dbrx)
- [LagLlama (time series forecasting)](https://github.com/time-series-foundation-models/lag-llama)
- [DiagnosisGPT 6B/34B (medical diagnosis LLM)](https://github.com/FreedomIntelligence/Chain-of-Diagnosis)
- [AI21 Jamba 1.5 12B/94B (hybrid SSM-Transformer)](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini)
- [Microsoft's GRIN-MoE](https://huggingface.co/microsoft/GRIN-MoE)
- [Athene V2 72B (Qwen Fine-Tune)](https://huggingface.co/collections/Nexusflow/athene-v2-6735b85e505981a794fb02cc)
- [LG's EXAONE-3.5 (several sizes)](https://huggingface.co/collections/LGAI-EXAONE/exaone-35-674d0e1bb3dcd2ab6f39dbb4)
- [Nvidia's Llama-3.1-Nemotron-70B](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)
- [Gemma 27B (exl2 format)](https://huggingface.co/mo137/gemma-2-27b-it-exl2)
- [Tencent's Hunyuan Large MoE](https://huggingface.co/tencent/Tencent-Hunyuan-Large)
- [AllenAI's Tulu 3 (Llama3.1 Fine-Tune)](https://huggingface.co/collections/allenai/tulu-3-models-673b8e0dc3512e30e7dc54f5)
- [Jamba 1.6 (hybrid SSM-Transformer)](https://huggingface.co/collections/ai21labs/jamba-16-67c990671a26dcbfa62d18fa)
- [Cogito v1 Preview (hybrid reasoning)](https://huggingface.co/collections/deepcogito/cogito-v1-preview-67eb105721081abe4ce2ee53)
- [GLM-4-32B](https://huggingface.co/matteogeniaccio/GLM-4-32B-0414-GGUF-fixed/tree/main)
- [Unsloth Dynamic Quants](https://huggingface.co/collections/unsloth/unsloth-dynamic-20-quants-68060d147e9b9231112823e6)


### Smaller model releases (model series and derivatives)

#### Embedding / Encoder Models
- [Jina Embeddings v3](https://huggingface.co/jinaai/jina-embeddings-v3)
- [Nomic Embeddings v2 MoE](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
- [IBM Granite Embeddings](https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb)
- [EuroBERT](https://huggingface.co/blog/EuroBERT/release)

#### Microsoft
- [phi-2 2.7B](https://huggingface.co/microsoft/phi-2)
- [phi-3 / phi-3.5](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)
- [phi-3.5 uncensored](https://huggingface.co/bartowski/Phi-3.5-mini-instruct_Uncensored-GGUF)
- [phi-4](https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4)
- [Unsloth-phi-4](https://huggingface.co/unsloth/phi-4-GGUF)
- [Glider (Phi 3.5 Mini based text evaluation model)](https://huggingface.co/PatronusAI/glider)
- [Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning)

#### Google
- [Gemma 2B and 7B](https://huggingface.co/blog/gemma)
- [Gemma2 9B](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF)
- [CodeGemma](https://www.kaggle.com/models/google/codegemma)
- [RecurrentGemma](https://huggingface.co/google/recurrentgemma-9b)
- [TigerGemma 9B v3 (fine-tune)](https://huggingface.co/TheDrummer/Tiger-Gemma-9B-v3)
- [SEA-LION (South East Asia fine-tune)](https://huggingface.co/aisingapore/gemma2-9b-cpt-sea-lionv3-base)
- [Gemma with questions (fine-tune)](https://huggingface.co/prithivMLmods/GWQ-9B-Preview2)

#### Mistral (feat. Nvidia)
- NeMo 12B
  - https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
  - https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct
- NeMo Minitron (pruned, distilled, quantized)
  - [4B](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct)
  - [8B](https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base)
- [Ministral 8B Instruct](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)
- [Mistral-Small-24B-Instruct-2501 (aka Mistral Small 3)](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
- [DeepHermes-3-Mistral](https://huggingface.co/collections/NousResearch/deephermes-67d2ff8c9246cc09a7bd8add)

#### Apple
- [Open-ELM](https://github.com/apple/corenet/tree/main/mlx_examples/open_elm)
- [Core ML Gallery of on-device models](https://huggingface.co/apple)

#### Hugging Face
- [SmolLM v1](https://huggingface.co/blog/smollm)
- [SmolLM v2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)

#### Zyphra
- [Zamba2-2.7B](https://huggingface.co/Zyphra/Zamba2-2.7B)
- [Zamba2-1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B)
- [Zamba2-7B](https://huggingface.co/Zyphra/Zamba2-7B-Instruct)

#### Meta
- [Llama 3.2 (collection of small models and big vision models)](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
- [MobileLLM](https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95)
- [Llama3 8B LexiFun Uncensored V1 (fine-tune)](https://huggingface.co/Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1)
- [Selene-1-Mini-Llama-3.1-8B (Llama-based small evaluation and scoring LLM)](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B)
- [DeepHermes-3-Llama](https://huggingface.co/collections/NousResearch/deephermes-67d2ff8c9246cc09a7bd8add)

#### WizardLM Uncensored
- [13B](https://huggingface.co/cognitivecomputations/WizardLM-13B-Uncensored)
- [7B](https://huggingface.co/cognitivecomputations/WizardLM-7B-Uncensored)

#### Cohere
- [Aya 23 8B/35B (multilingual specialized)](https://huggingface.co/collections/CohereForAI/c4ai-aya-23-664f4cda3fa1a30553b221dc)
- [Command R7B](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024)

#### Alibaba (Qwen plus merges / fine-tunes)
- [Qwen-2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M)
- SuperNova-Medius 14B (distillation merge between Qwen and Llama)
  - https://huggingface.co/arcee-ai/SuperNova-Medius
  - https://huggingface.co/bartowski/SuperNova-Medius-GGUF
- [SmallThinker-3B-Preview](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview)
- [Confucius-o1-14B (Qwen reasoning fine-tune)](https://huggingface.co/netease-youdao/Confucius-o1-14B)
- [Chirp 3B](https://huggingface.co/ozone-research/Chirp-01)
- [UIGEN 14B (based on Qwen 2.5 14B Coder)](https://huggingface.co/smirki/UIGEN-T1.1-Qwen-14B)
- [Rombo LLM Qwen Merge](https://huggingface.co/bartowski/Rombo-Org_Rombo-LLM-V3.1-QWQ-32b-GGUF)

#### Deepseek
- [R1 distillations](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d)
- [Arcee Meastro 7B (distillation fine-tune)](https://huggingface.co/arcee-ai/Arcee-Maestro-7B-Preview)

#### IBM
- [Granite Code Models](https://github.com/ibm-granite/granite-code-models)
- [Granite v3.1](https://huggingface.co/collections/ibm-granite/granite-31-language-models-6751dbbf2f3389bec5c6f02d)
- [Granite v3.2](https://huggingface.co/collections/ibm-granite/granite-32-language-models-67b3bc8c13508f6d064cff9a)
- [Granite v3.3](https://huggingface.co/collections/ibm-granite/granite-33-language-models-67f65d0cca24bcbd1d3a08e3)
- [Granite v4 Preview](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c)
- [Granite Experiments](https://huggingface.co/collections/ibm-granite/granite-experiments-6724f4c225cd6baf693dbb7a)

#### Other
- [Awesome Small Language Models list](https://github.com/slashml/awesome-small-language-models)
- [LLMs for on-device deployment](https://github.com/NexaAI/Awesome-LLMs-on-device)
- [MaLA 7B](https://huggingface.co/MaLA-LM/mala-500)
- [Yi-9B](https://huggingface.co/01-ai/Yi-9B)
- [Yi-Coder 9B](https://huggingface.co/01-ai/Yi-Coder-9B-Chat)
- [AutoCoder 7B](https://github.com/bin123apple/AutoCoder)
- [Replit Code v1.5 3B for coding](https://huggingface.co/replit/replit-code-v1_5-3b)
- [RWKV-5 Eagle 7B](https://huggingface.co/RWKV/v5-Eagle-7B-pth)
- [RefuelLLM-2 8B (data labeling model)](https://huggingface.co/refuelai/Llama-3-Refueled)
- [Doctor Dignity 7B](https://github.com/llSourcell/Doctor-Dignity)
- [Prometheus 2 7B/8x7B (LLM for LLM evaluation)](https://github.com/prometheus-eval/prometheus-eval)
- [Prem-1B (RAG expert model)](https://blog.premai.io/introducing-prem-1b/)
- [CodeGeeX4 9B](https://huggingface.co/THUDM/codegeex4-all-9b)
- [BigTranslate 13B](https://github.com/ZNLP/BigTranslate)
- [Starcoder 2 3B/7B/15B](https://github.com/bigcode-project/starcoder2)
- [ContextualAI's OLMoE](https://contextual.ai/olmoe-mixture-of-experts)
- [OpenCoder 1.5B/8B](https://huggingface.co/collections/infly/opencoder-672cec44bbb86c39910fb55e)
- [NuExtract 1.5](https://huggingface.co/collections/numind/nuextract-15-670900bc74417005409a8b2d)
- [Kurage Multilingual 7B](https://huggingface.co/lightblue/kurage-multilingual)
- [Teuken 7B (based on European languages)](https://huggingface.co/openGPT-X/Teuken-7B-instruct-commercial-v0.4)
- [EuroLLM 9B](https://huggingface.co/utter-project/EuroLLM-9B)
- [Moxin 7B](https://github.com/moxin-org/Moxin-LLM)
- [Megrez 3B Instruct](https://huggingface.co/Infinigence/Megrez-3B-Instruct)
- [GRMR v2.0 (grammar checking)](https://huggingface.co/collections/qingy2024/grmr-v20-6759d4172e557af98a2feabc)
- [Falcon 3 (several sizes)](https://huggingface.co/blog/falcon3)
- [AMD's OLMo 1B](https://huggingface.co/amd/AMD-OLMo)
- [Fox-1-1.6B-Instruct](https://huggingface.co/tensoropera/Fox-1-1.6B-Instruct-v0.1)
- [Pleias models (trained mainly on common corpus)](https://huggingface.co/collections/PleIAs/common-models-674cd0667951ab7c4ef84cc4)
- [AllenAI's OLMo 2](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc)
- [InternLM3 8B Instruct](https://huggingface.co/internlm/internlm3-8b-instruct)
- Recommended for translation
  - [Meta's NLLB](https://huggingface.co/facebook/nllb-200-3.3B)
  - [ALMA](https://github.com/fe1ixxu/ALMA)
  - [Unbabel's Towerbase](https://huggingface.co/Unbabel/TowerBase-13B-v0.1)
  - [Google's MADLAD-400-10B-MT](https://huggingface.co/google/madlad400-10b-mt )
- [Arch-Function-3B (function calling)](https://huggingface.co/katanemo/Arch-Function-3B)
- [Reka-Flash-3 21B](https://huggingface.co/RekaAI/reka-flash-3)
- [DeepCoder-14B-Preview](https://huggingface.co/agentica-org/DeepCoder-14B-Preview)
- [XiaomiMiMo (reasoning)](https://github.com/XiaomiMiMo/MiMo)
- [JetBrains Mellum-4b-base (coding)](https://huggingface.co/JetBrains/Mellum-4b-base)
- [Helium 1 (multilingual and modular)](https://huggingface.co/collections/kyutai/helium-1-681237bbba8c1cf18a02e4bd)


### Multimodal / Vision models

#### LLaVA
- [Github](https://github.com/haotian-liu/LLaVA)
- [First Impressions with LLaVA 1.5](https://blog.roboflow.com/first-impressions-with-llava-1-5/)
- [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
- [LLaVA-OneVision](https://huggingface.co/collections/lmms-lab/llava-onevision-66a259c3526e15166d6bba37)

#### Alibaba
- [Qwen-VL](https://qwenlm.github.io/blog/qwen-vl/)
- Qwen-VL2
  - https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
  - https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct
- [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5)
- [Qwen2-Audio](https://huggingface.co/collections/Qwen/qwen2-audio-66b628d694096020e0c52ff6)
- [VideoLLaMA3 (video understanding)](https://huggingface.co/collections/DAMO-NLP-SG/videollama3-678cdda9281a0e32fe79af15)

#### DeepSeek
- [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)
- [Janus 1.3B](https://huggingface.co/deepseek-ai/Janus-1.3B)
- DeepSeek-VL2 (MoE)
  - [Github](https://github.com/deepseek-ai/DeepSeek-VL2)
  - [HuggingFace](https://huggingface.co/collections/deepseek-ai/deepseek-vl2-675c22accc456d3beb4613ab)
- [Janus Pro 7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)

#### Google
- [PaliGemma](https://www.kaggle.com/models/google/paligemma)
- [PaliGemma 2](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48)
- [Inksight (Hand-writing to text)](https://github.com/google-research/inksight/)

#### Microsoft
- [Florence](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de)
- [ComfyUI Florence2 Workflow](https://github.com/kijai/ComfyUI-Florence2)
- [Phi-3-vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cuda)
- [Florence-VL](https://huggingface.co/jiuhai)
- [Phi-4](https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4)
- [Magma 8B (agentic multimodal)](https://huggingface.co/microsoft/Magma-8B)

#### OpenBMB
- [MiniCPM V2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)
- [MiniCPM V2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)

#### Nvidia
- [Eagle](https://huggingface.co/NVEagle/Eagle-X5-13B)
- [NVLM-D](https://huggingface.co/nvidia/NVLM-D-72B)

#### Mistral
- [Pixtral 12B](https://huggingface.co/mistralai/Pixtral-12B-2409)
- [Pixtral 124B](https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411)

#### Meta
- [Llama 3.2-Vision](https://huggingface.co/collections/meta-llama/)
- [Apollo 7B (based on Qwen, taken down by Meta)](https://huggingface.co/GoodiesHere/Apollo-LMMs-Apollo-7B-t32)

#### Moondream (vision model on edge devices)
- https://github.com/vikhyat/moondream
- https://huggingface.co/vikhyatk/moondream2
- [Moondream 2B (small vlm)](https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit)

#### Other
- [Apple's Ferret](https://github.com/apple/ml-ferret)
- [Idefics2 8B](https://huggingface.co/HuggingFaceM4/idefics2-8b)
- [Mini-Omni](https://huggingface.co/gpt-omni/mini-omni)
- [Ultravox](https://github.com/fixie-ai/ultravox)
- [Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
- [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
- OCR
  - [GOT OCR 2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/)
  - Ovis 1.6 (vision model with LLM based on Gemma 2 9B)
    - https://github.com/Ucas-HaoranWei/GOT-OCR2.0/
    - https://huggingface.co/stepfun-ai/GOT-OCR2_0
  - [Ovis2 collection (strong with OCR)](https://huggingface.co/collections/AIDC-AI/ovis2-67ab36c7e497429034874464)
  - [olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview)
- [Aria](https://huggingface.co/rhymes-ai/Aria)
- [OpenGVLab's InternVL 2.5](https://huggingface.co/collections/OpenGVLab/internvl-25-673e1019b66e2218f68d7c1c)
- [Megrez 3B Omni](https://huggingface.co/Infinigence/Megrez-3B-Omni)
- [YuLan Mini](https://github.com/RUC-GSAI/YuLan-Mini)
- [Bytedance's Sa2VA](https://huggingface.co/collections/ByteDance/sa2va-model-zoo-677e3084d71b5f108d00e093)
- [LlamaV o1 11B Vision Instruct](https://huggingface.co/omkarthawakar/LlamaV-o1)
- [IBM's Granite v3.2](https://huggingface.co/collections/ibm-granite/granite-vision-models-67b3bd4ff90c915ba4cd2800)
- [Huggingface's SmolVLM2](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)
- [Cohere's Aya Vision 8B & 32B](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision-67c4ccd395ca064308ee1484)
- [Kimi-VL-A3B](https://huggingface.co/collections/moonshotai/kimi-vl-a3b-67f67b6ac91d3b03d382dd85)
