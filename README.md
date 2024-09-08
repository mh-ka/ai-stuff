# TOC
[AI Stuff](#ai-stuff)
  * [LLMs](#llms)
    + [Basics](#basics)
    + [Models](#models)
    + [Tools around using LLMs](#tools-around-using-llms)
    + [AI Developer Topics](#ai-developer-topics)
    + [Evaluation](#evaluation)
  * [Audio](#audio)
    + [Voicecraft](#voicecraft)
    + [Meta](#meta)
    + [speech2txt](#speech2txt)
    + [txt2speech and txt2audio](#txt2speech-and-txt2audio)
    + [Music production](#music-production)
    + [Other](#other)
  * [Image and Video and 3D generation](#image-and-video-and-3d-generation)
    + [Stable diffusion](#stable-diffusion)
    + [Flux](#flux)
    + [Animation and Video etc](#animation-and-video-etc)
  * [Reports and Articles and Sources](#reports-and-articles-and-sources)
    + [Reports](#reports)
    + [Articles](#articles)
    + [News sources](#news-sources)

<sup><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i><sup>


# AI Stuff


## LLMs

### Basics
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

<sup><sub>[back to top](#toc)</sub></sup>

### Models
- Meta
  - Llama
    - CodeLLama: https://github.com/facebookresearch/codellama
    - slowllama: with offloading to SSD/Memory: https://github.com/okuvshynov/slowllama
    - Llama 3: https://llama.meta.com/llama3/
  - NLLB (No language left behind): https://github.com/facebookresearch/fairseq/tree/nllb
- Mistral
  - Finetune Mistral on your own data: https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb
  - Models on huggingface: https://huggingface.co/mistralai (Codestral, Mathstral, Nemo, Mixtral, Mistral Large etc.)
- DeepSeek
  - DeepSeek-LLM 67B: https://github.com/deepseek-ai/DeepSeek-LLM
  - DeepSeek-V2-Chat: https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat
- Alibaba
  - Qwen 1.5: https://qwenlm.github.io/blog/qwen1.5/
  - CodeQwen1.5-7B-Chat
    - HuggingFace page: https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat
    - Discussion: https://www.reddit.com/r/LocalLLaMA/comments/1c6ehct/codeqwen15_7b_is_pretty_darn_good_and_supposedly/
  - Qwen2: https://qwenlm.github.io/blog/qwen2/
- Misc
  - Phind-CodeLlama-34B v2: https://huggingface.co/Phind/Phind-CodeLlama-34B-v2
  - Doctor Dignity: https://github.com/llSourcell/Doctor-Dignity
  - BigTranslate: https://github.com/ZNLP/BigTranslate
  - MaLA: https://huggingface.co/MaLA-LM/mala-500
  - Prometheus 2 (LLM for LLM evaluation): https://github.com/prometheus-eval/prometheus-eval
  - LLM360's K2 (fully open source): https://huggingface.co/LLM360/K2
  - Starcoder 2: https://github.com/bigcode-project/starcoder2
  - DBRX Base and Instruct: https://github.com/databricks/dbrx
  - Cohere
    - Command-R: https://huggingface.co/CohereForAI/c4ai-command-r-v01
    - Command-R 08-2024: https://huggingface.co/bartowski/c4ai-command-r-08-2024-GGUF
  - Snowflake Arctic: https://github.com/Snowflake-Labs/snowflake-arctic
  - IBM's Granite: https://github.com/ibm-granite/granite-code-models
  - AutoCoder: https://github.com/bin123apple/AutoCoder
  - LagLlama (time series forecasting): https://github.com/time-series-foundation-models/lag-llama
  - Ollama library: https://ollama.com/library
  - DiagnosisGPT (medical diagnosis LLM): https://github.com/FreedomIntelligence/Chain-of-Diagnosis
  - AI21 Jamba 1.5 Mini (hybrid SSM-Transformer): https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini
  - Yi-Coder 9B: https://huggingface.co/01-ai/Yi-Coder-9B-Chat 
- Small sized
  - Microsoft
    - phi-2 2.7B: https://huggingface.co/microsoft/phi-2
    - phi-3 / phi-3.5: https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3
  - Replit Code v1.5 3B for coding: https://huggingface.co/replit/replit-code-v1_5-3b
  - RWKV-5 Eagle 7B: https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers
  - Google
    - Gemma 2B and 7B: https://huggingface.co/blog/gemma
    - Gemma2 9B: https://huggingface.co/bartowski/gemma-2-9b-it-GGUF
    - CodeGemma: https://www.kaggle.com/models/google/codegemma
    - RecurrentGemma: https://huggingface.co/google/recurrentgemma-9b
  - Moondream (vision model on edge devices)
    - https://github.com/vikhyat/moondream
    - https://huggingface.co/vikhyatk/moondream2
  - Yi-9B: https://huggingface.co/01-ai/Yi-9B
  - Mistral (feat. Nvidia)
    - NeMo 12B:
      - https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
      - https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct
    - NeMo Minitron 8B (pruned & distilled)
      - https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base
  - Apple
    - Open-ELM: https://github.com/apple/corenet/tree/main/mlx_examples/open_elm
    - Core ML Gallery of on-device models: https://huggingface.co/apple
  - RefuelLLM-2 (data labeling model): https://www.refuel.ai/blog-posts/announcing-refuel-llm-2
  - Prem-1B (RAG expert model): https://blog.premai.io/introducing-prem-1b/
  - Cohere's Aya 23 (multilingual specialized): https://huggingface.co/collections/CohereForAI/c4ai-aya-23-664f4cda3fa1a30553b221dc
  - Mistral's 7B-Instruct-v0.3: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
  - Llama3 8B LexiFun Uncensored V1: https://huggingface.co/Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1
  - Hugging Face's SmolLM: https://huggingface.co/blog/smollm
  - Zyphra
    - Zamba2-2.7B: https://huggingface.co/Zyphra/Zamba2-2.7B
    - Zamba2-1.2B: https://huggingface.co/Zyphra/Zamba2-1.2B
  - LLMs for on-device deployment: https://github.com/NexaAI/Awesome-LLMs-on-device/tree/main
  - ContextualAI's OLMoE: https://contextual.ai/olmoe-mixture-of-experts
- Multimodal
  - LLaVA: https://github.com/haotian-liu/LLaVA
    - First Impressions with LLaVA 1.5: https://blog.roboflow.com/first-impressions-with-llava-1-5/
    - LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/
    - LLaVA-OneVision: https://huggingface.co/collections/lmms-lab/llava-onevision-66a259c3526e15166d6bba37
  - Apple's Ferret: https://github.com/apple/ml-ferret
  - Alibaba
    - Qwen-VL: https://qwenlm.github.io/blog/qwen-vl/
    - Qwen-VL2: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
  - DeepSeek-VL: https://github.com/deepseek-ai/DeepSeek-VL
  - Idefics2 8B: https://huggingface.co/HuggingFaceM4/idefics2-8b
  - Google's PaliGemma: https://www.kaggle.com/models/google/paligemma
  - YOLOv10 (Real-time object detection): https://github.com/THU-MIG/yolov10 (https://github.com/ultralytics/ultralytics)
  - Microsoft's Florence 2 (non-OCR vision tasks): https://huggingface.co/microsoft/Florence-2-large
  - Microsoft's Phi-3-vision: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cuda
  - OpenBMB's MiniCPM-Llama3-V 2.5: https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
  - Ultravox: https://github.com/fixie-ai/ultravox
  - Nvidia's Eagle: https://huggingface.co/NVEagle/Eagle-X5-13B
  - Mini-Omni: https://huggingface.co/gpt-omni/mini-omni
- List of relevant European companies in LLM area: DeepL, Mistral, Silo AI, Aleph Alpha

<sup><sub>[back to top](#toc)</sub></sup>

### Tools around using LLMs
- Local LLM hosting
  - vLLM (local python library for running models): https://docs.vllm.ai/en/latest/getting_started/quickstart.html
  - LMStudio (local model hosting): https://lmstudio.ai/
  - Nvidia's ChatRTX (local chat with files and image search): https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/
  - Jan app (local model hosting): https://jan.ai/
  - Leeroo Orchestrator (choose model depending on task): https://github.com/leeroo-ai/leeroo_orchestrator
  - FreedomGPT (local model hosting): https://www.freedomgpt.com/
  - ExLlama2 (optimized local inference): https://github.com/turboderp/exllamav2
  - pinokio (Simple app for running many AI tools locally): https://pinokio.computer/
  - GPT4All (local LLM hosting): https://github.com/nomic-ai/gpt4all
  - h2oGPT (Private Q&A with your own documents): https://github.com/h2oai/h2ogpt
  - chatd (chat with your own documents): https://github.com/BruceMacD/chatd
  - Nextcloud AI assistant: https://docs.nextcloud.com/server/latest/admin_manual/ai/index.html (https://github.com/nextcloud/all-in-one/tree/main)
  - Opera One Developer (browser with LLMs): https://blogs.opera.com/news/2024/04/ai-feature-drops-local-llms/
  - LocalAI (local server hosting): https://localai.io/
  - LLM CLI tool: https://llm.datasette.io/en/stable/
  - Run Llama3 70B locally: https://simonwillison.net/2024/Apr/22/llama-3/
  - Perplexica: https://github.com/ItzCrazyKns/Perplexica
  - Local OSS search engine: https://github.com/Nutlope/turboseek
  - LibreChat: https://github.com/danny-avila/LibreChat
  - OpenPerplex: https://github.com/YassKhazzan/openperplex_backend_os
- Local coding assistance
  - Open Interpreter (local model hosting with code execution and file input): https://github.com/KillianLucas/open-interpreter
  - Tabby (Self-hosted coding assistant): https://tabby.tabbyml.com/docs/getting-started
  - Sweep (AI junior developer): https://github.com/sweepai/sweep
  - GPT-engineer (let GPT write and run code): https://github.com/gpt-engineer-org/gpt-engineer
  - OSS alternative to Devin AI Software Engineer: https://github.com/stitionai/devika
  - OpenDevin: https://github.com/OpenDevin/OpenDevin
  - VSCode plugin with Ollama for local coding assistant: https://ollama.com/blog/continue-code-assistant
- More coding assistance
  - Aider (coding): https://github.com/paul-gauthier/aider
  - Cursor (GPT-API-based code editor app): https://cursor.sh/
  - Plandex (GPT-API-based coding agent): https://github.com/plandex-ai/plandex
  - SWE-agent (another GPT-API-based coding agent): https://github.com/princeton-nlp/SWE-agent
  - OpenUI (another GPT-API-based coding agent for web components): https://github.com/wandb/openui
  - Tabnine: https://www.tabnine.com/
  - Melty: https://github.com/meltylabs/melty
- Phind (AI based search engine): https://www.phind.com/search?home=true
- Scalene (High Performance Python Profiler): https://github.com/plasma-umass/scalene
- Lida (visualization and infographics generated with LLMs): https://github.com/microsoft/lida
- OpenCopilot (LLM integration with provided Swagger APIs): https://github.com/openchatai/OpenCopilot
- LogAI (log analytics): https://github.com/salesforce/logai
- Jupyter AI (jupyter notebook AI assistance): https://github.com/jupyterlab/jupyter-ai
- LLM CLI Tool: https://llm.datasette.io/en/stable/
- OCR Tesseract tool (running in browser locally): https://github.com/simonw/tools/tree/main
- LLM cmd Assistant: https://github.com/simonw/llm-cmd
- Semgrep (autofix using LLMs): https://choly.ca/post/semgrep-autofix-llm/
- GPT Investor: https://github.com/mshumer/gpt-investor
- AI Flow (connect different models): https://ai-flow.net
- Model playgrounds
  - Poe: https://poe.com/login
  - Qolaba: https://www.qolaba.ai/
  - ChatLLM: https://chatllm.abacus.ai/
- Shopping assistant
  - https://www.claros.so/

<sup><sub>[back to top](#toc)</sub></sup>

### AI Developer Topics
- Training and Fine-Tuning
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
- Integrating, hosting, merging LLMs 
  - RAG
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
  - Data extraction:
    - Pre-processing unstructured data: https://github.com/Unstructured-IO/unstructured
    - Crawl4AI (simplify crawling and data extraction as input for LLMs): https://github.com/unclecode/crawl4ai
    - URL to LLM input converter: https://github.com/jina-ai/reader
    - LLM Scraper (extract structured data from URL): https://github.com/mishushakov/llm-scraper
    - Datachain (python library for unstructured data processing): https://github.com/iterative/datachain
    - ThePipe (markdown and visuals extraction from PDFs, URLs, etc.): https://github.com/emcf/thepipe
    - Firecrawl (websites2markdown): https://github.com/mendableai/firecrawl
  - Docker GenAI Stack: https://github.com/docker/genai-stack
  - LLM App Stack: https://github.com/a16z-infra/llm-app-stack
  - Prem AI infrastructure tooling: https://github.com/premAI-io/prem-app
  - Langchain Templates (templates for building AI apps): https://github.com/langchain-ai/langchain/tree/master/templates
  - Dify (local LLM app development platform): https://github.com/langgenius/dify
  - Mergekit (framework for merging LLMs): https://github.com/arcee-ai/mergekit
  - Langwatch (LLM monitoring tools suite): https://github.com/langwatch/langwatch
  - Transformer.js (browser-based AI inference): https://github.com/xenova/transformers.js
  - ExecuTorch (On-Device AI framework): https://pytorch.org/executorch/stable/intro-overview.html
  - Code Interpreter SDK (sandbox for LLM code execution): https://github.com/e2b-dev/code-interpreter
  - Ratchet (ML developer toolkit for web browser inference): https://github.com/huggingface/ratchet
  - Phi-3
    - Phi-3 on device: https://huggingface.co/blog/Emma-N/enjoy-the-power-of-phi-3-with-onnx-runtime
    - Microsoft's Cookbook on using Phi-3: https://github.com/microsoft/Phi-3CookBook
  - Build private research assistant using llamaindex and llamafile: https://www.llamaindex.ai/blog/using-llamaindex-and-llamafile-to-build-a-local-private-research-assistant
  - Selfie (personalized local text generation): https://github.com/vana-com/selfie
  - Empirical Study of LLaMA3 Quantization: https://arxiv.org/abs/2404.14047
- Agents
  - How to design an agent for production: https://blog.langchain.dev/how-to-design-an-agent-for-production/
  - Document-Oriented Agents: https://towardsdatascience.com/document-oriented-agents-a-journey-with-vector-databases-llms-langchain-fastapi-and-docker-be0efcd229f4
  - Experts.js (Multi AI Agent Systems Framework in Javascript): https://github.com/metaskills/experts
  - Pipecat (build conversational agents): https://github.com/pipecat-ai/pipecat
  - AI Agent Infrastructure: https://www.madrona.com/the-rise-of-ai-agent-infrastructure/
  - Qwen Agent framework: https://github.com/QwenLM/Qwen-Agent
  - LLM Pricing: https://huggingface.co/spaces/philschmid/llm-pricing
- Misc
  - List of AI developer tools: https://github.com/sidhq/YC-alum-ai-tools
  - LLMs and political biases: https://www.technologyreview.com/2023/08/07/1077324/ai-language-models-are-rife-with-political-biases
  - Using LLM to create a audio storyline: https://github.com/Audio-AGI/WavJourney
  - Open questions for AI Engineering: https://simonwillison.net/2023/Oct/17/open-questions/#slides-start
  - AI for journalism use cases: https://www.youtube.com/watch?v=BJxPKr6ixSM
  - AI Grant (accelerator program for AI startups with list of applicants): https://aigrant.com/
  - Reducing costs and improving performance using LLMs: https://portkey.ai/blog/implementing-frugalgpt-smarter-llm-usage-for-lower-costs
  - What We Learned from a Year of Building with LLMs:
    - https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i
    - https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii
  - Showcase GPT-2 re-implementation in Rust: https://github.com/felix-andreas/gpt-burn
  - Reproducing GPT-2 (124M) in llm.c:
    - Text: https://github.com/karpathy/llm.c/discussions/481
    - Video: https://www.youtube.com/watch?v=l8pRSuU81PU
  - Reproducing GPT-2 in Excel spreadsheet: https://spreadsheets-are-all-you-need.ai/
  - Pruning Approach for LLMs (Meta, Bosch): https://github.com/locuslab/wanda
  - How to get great AI code completions (technical insights on code completion): https://sourcegraph.com/blog/the-lifecycle-of-a-code-ai-completion
  - AI NPCs in games: https://huggingface.co/blog/npc-gigax-cubzh
  - llama.ttf (using font engine to host a LLM): https://fuglede.github.io/llama.ttf
  - RouteLLM framework: https://github.com/lm-sys/RouteLLM
  - Reverse engineering of Github Copilot extension: https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals.html
  - LeRobot: https://github.com/huggingface/lerobot

<sup><sub>[back to top](#toc)</sub></sup>

### Evaluation
- How to construct domain-specific LLM evaluation systems: https://hamel.dev/blog/posts/evals/
- Big Code Models Leaderboard: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
- Chatbot Arena: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
- Zephyr 7B infos on reddit: https://www.reddit.com/r/LocalLLaMA/comments/17hjgdg/zephyr_7b_beta_a_new_mistral_finetune_is_out/
- Coding Benchmark: https://prollm.toqan.ai/leaderboard
- CanAiCode Leaderboard: https://huggingface.co/spaces/mike-ravkine/can-ai-code-results
- Vision Language Model Leaderboard: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
- OpenCompass 2.0: https://github.com/open-compass/opencompass
- Open LLM Progress Tracker: https://huggingface.co/spaces/andrewrreed/closed-vs-open-arena-elo
- MMLU Pro: https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro
- LLM API Performance Leaderboard: https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard
- SEAL Leaderboard: https://scale.com/leaderboard
- Lightweight Library from OpenAI to evaluate LLMs: https://github.com/openai/simple-evals
- Open LLM Leaderboard 2: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- Skeleton Key Jailbreak: https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique
- LLM Eval simplified: https://www.philschmid.de/llm-evaluation
- LiveBench: https://livebench.ai/
- Benchmark Aggregator: https://benchmark-aggregator-lvss.vercel.app/

<sup><sub>[back to top](#toc)</sub></sup>


## Audio

### Voicecraft
- Repo: https://github.com/jasonppy/VoiceCraft
- Huggingface: https://huggingface.co/pyp1/VoiceCraft/tree/main
- Docker version: https://github.com/pselvana/VoiceCrafter
- API version: https://github.com/lukaszliniewicz/VoiceCraft_API

<sup><sub>[back to top](#toc)</sub></sup>

### Meta
- MusicGen: https://huggingface.co/facebook/musicgen-stereo-melody-large
  - Trying out MusicGen: https://til.simonwillison.net/machinelearning/musicgen
- SeamlessStreaming: https://huggingface.co/facebook/seamless-streaming
  - Seamless Framework: https://github.com/facebookresearch/seamless_communication
- MAGNeT (txt-2-music/-audio): https://analyticsindiamag.com/meta-launches-magnet-an-open-source-text-to-audio-model/
  - Hugging Face page with models: https://huggingface.co/collections/facebook/magnet-659ef0ceb62804e6f41d1466

<sup><sub>[back to top](#toc)</sub></sup>

### speech2txt
- OpenAI's Whisper: https://github.com/openai/whisper
  - Distil-Whisper: https://github.com/huggingface/distil-whisper/issues/4
  - Insanely fast whisper: https://github.com/Vaibhavs10/insanely-fast-whisper
  - WhisperKit for Apple devices: https://www.takeargmax.com/blog/whisperkit
- Nvidia's Canary (with translation): https://nvidia.github.io/NeMo/blogs/2024/2024-02-canary/
- Qwen2-Audio-7B: https://huggingface.co/Qwen/Qwen2-Audio-7B
- Speech2Speech pipeline: https://github.com/huggingface/speech-to-speech

<sup><sub>[back to top](#toc)</sub></sup>

### txt2speech and txt2audio
- XTTS v1: https://huggingface.co/coqui/XTTS-v1
- AudioLDM2: https://github.com/haoheliu/audioldm2
- Bark: https://github.com/suno-ai/bark
- Tracker page for open access text2speech models: https://github.com/Vaibhavs10/open-tts-tracker
- MetaVoice: https://github.com/metavoiceio/metavoice-src
- Pheme TTS framework: https://github.com/PolyAI-LDN/pheme
- OpenAI TTS: https://platform.openai.com/docs/guides/text-to-speech
- TTS Arena Leaderboard: https://huggingface.co/spaces/TTS-AGI/TTS-Arena
- OpenVoice: https://github.com/myshell-ai/OpenVoice
- Stable Audio Open: https://huggingface.co/stabilityai/stable-audio-open-1.0
- MARS5-TTS: https://github.com/Camb-ai/MARS5-TTS
- Alibaba's FunAudioLLM framework (includes CosyVoice & SenseVoice): https://github.com/FunAudioLLM
- MeloTTS: https://github.com/myshell-ai/MeloTTS
- Parler TTS: https://github.com/huggingface/parler-tts
- WhisperSpeech: https://github.com/collabora/WhisperSpeech

<sup><sub>[back to top](#toc)</sub></sup>

### Music production
- LANDR mastering plugin: https://www.gearnews.de/landr-mastering-plugin/
- Drumloop.ai: https://www.gearnews.de/drumloop-ai-baut-euch-automatisch-beats-und-drumloops-durch-ki/

<sup><sub>[back to top](#toc)</sub></sup>

### Other
- LAION AI Voice Assistant BUD-E: https://github.com/LAION-AI/natural_voice_assistant
- AI Language Tutor:
  - https://www.univerbal.app/
  - https://yourteacher.ai/
- Speech Note Offline STT, TTS and Machine Translation: https://github.com/mkiol/dsnote
- DenseAV (locates sound and learns meaning of words): https://github.com/mhamilton723/DenseAV

<sup><sub>[back to top](#toc)</sub></sup>


## Image and Video and 3D generation

### Stable diffusion
- Models
  - Stable Video Diffusion 1.1: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
  - Stable Cascade: https://stability.ai/news/introducing-stable-cascade
  - SDXL ControlNet 1.0 Models: https://huggingface.co/xinsir
    - Scribble: https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0
  - SDXL-based Fine-tuned models
    - DreamShaper XL Turbo v2: https://civitai.com/models/112902/dreamshaper-xl
    - Animagine XL 3.0: https://huggingface.co/cagliostrolab/animagine-xl-3.0
    - Nebul.Redmond: https://civitai.com/models/269988
    - Pixel Art Diffusion XL: https://civitai.com/models/277680/pixel-art-diffusion-xl
    - Juggernaut XL v7: https://civitai.com/models/133005?modelVersionId=240840
    - Juggernaut XL v9: https://huggingface.co/RunDiffusion/Juggernaut-XL-v9
    - ThinkDiffusionXL: https://civitai.com/models/169868/thinkdiffusionxl
    - Realism Engine SDXL: https://civitai.com/models/152525
    - Amiga Style: https://civitai.com/models/297955/amiga-style
    - Cheyenne Comic Model: https://civitai.com/models/198051/cheyenne?modelVersionId=318969
    - LEOSAM AIArt: https://civitai.com/models/219791?modelVersionId=247832
    - Artfully Echelier: https://civitai.com/models/391436/artfullyechelier-sdxl-v1
    - CineVisionXL: https://civitai.com/models/188208?modelVersionId=211388
    - Sahastrakoti XL: https://civitai.com/models/139489/sahastrakoti
  - SD 1.5 based Fine-tuned models
    - GhostMix: https://civitai.com/models/36520/ghostmix
    - ReV Animated: https://civitai.com/models/7371/rev-animated
    - EpicPhotoGasm: https://civitai.com/models/132632/epicphotogasm
    - Haveall: https://civitai.com/models/213692
    - Vivid Watercolors: https://civitai.com/models/4998/vivid-watercolors
    - Ghibli Diffusion: https://civitai.com/models/1066/ghibli-diffusion
    - UnstableInkDream: https://civitai.com/models/1540/unstableinkdream
    - Macro Diffusion: https://civitai.com/models/3863/macro-diffusion
    - helloFlatArt: https://civitai.com/models/183884/helloflatart?modelVersionId=206378
    - Mistoon Anime: https://civitai.com/models/24149/mistoonanime
    - fReel_Photo: https://civitai.com/models/137522/freelphoto
    - Rage Unleashed: https://civitai.com/models/115100/rage-unleashed
  - Segmind SSD 1B (Destilled SDXL): https://civitai.com/models/188863/segmind-ssd-1b
  - Realities Edge XL (SDXL-Turbo): https://civitai.com/models/129666/realities-edge-xl
  - Anime Model: https://huggingface.co/XpucT/Anime
  - Coloring Book: https://civitai.com/models/5153/coloring-book
  - RZ Vice Embedding: https://civitai.com/models/4920/rz-vice-21
  - Cool Japan Diffusion: https://huggingface.co/aipicasso/cool-japan-diffusion-2-1-1
  - Vintage Airbrushed: https://civitai.com/models/4291
  - Berlin Graffiti: https://huggingface.co/bakebrain/bergraffi-berlin-graffiti
  - Favorite Models: https://www.reddit.com/r/StableDiffusion/comments/19fdm0c/whats_your_favorite_model_currently/
- LORAs
  - SDXL
    - Remix XL: https://huggingface.co/Stkzzzz222/remixXL
    - iPhone Mirror Selfie: https://civitai.com/models/256058/iphone-mirror-selfie-or-sdxl
    - Aether Aqua: https://civitai.com/models/210754/aether-aqua-lora-for-sdxl
    - Claymation Style: https://civitai.com/models/208168/claymate-stop-motion-claymation-style-for-sdxl
    - Michael Craig Martin Style: https://civitai.com/models/488021/flat-colors-in-the-style-of-michael-craig-martin-sdxl-10
    - Retro Comics Style: https://civitai.com/models/491322/retro-comics-with-words-steven-rhodes-style-sdxl-10
    - Pastel: https://civitai.com/models/493874/pastel-photography-franck-bohbot-style-sdxl-10
    - Dreamworks: https://civitai.com/models/188622/essenz-how-to-train-your-dragon-3-the-hidden-world-dreamworks-style-lora-for-sdxl-10
    - Darkest Dungeon: https://civitai.com/models/188582/essenz-darkest-dungeon-chris-bourassa-style-lora-for-sdxl-10
    - Jean Giraud/Moebius: https://civitai.com/models/188660/essenz-jean-giraudmoebius-voyage-dhermes-style-lora-for-sdxl-10
    - Makoto Shinkai Anime: https://civitai.com/models/177512/essenz-makoto-shinkai-anime-screencap-your-name-weathering-with-you-suzume-style-lora-for-sdxl-10
    - Better Photography: https://civitai.com/models/198378/essenz-high-quality-heavily-post-processed-photography-style-lora-for-sdxl-10
    - Aether Watercolor & Ink: https://civitai.com/models/190242/aether-watercolor-and-ink-lora-for-sdxl
    - Jakub Rozalski: https://civitai.com/models/182205/essenz-jakub-rozalski-1920-scythe-iron-harvest-style-lora-for-sdxl-10
    - Ghibli: https://civitai.com/models/181883/essenz-nausicaa-of-the-valley-of-the-wind-anime-screencap-ghibli-hayao-myazaki-style-lora-for-sdxl-10
    - Another Ghibli: https://civitai.com/models/137562/s
    - MSPaint: https://civitai.com/models/183354?modelVersionId=205793
    - The Legend Of Korra: https://civitai.com/models/198368/essenz-the-legend-of-korra-screencap-avatar-style-lora-for-sdxl-10
    - Bubbles and Foam: https://civitai.com/models/170188/aether-bubbles-and-foam-lora-for-sdxl
    - Glitch: https://civitai.com/models/151542/aether-glitch-lora-for-sdxl
    - Loving Vincent: https://civitai.com/models/167991?modelVersionId=188927
    - Coloring Book: https://civitai.com/models/136348
    - Blacklight Makeup: https://civitai.com/models/134643
    - Stickers: https://civitai.com/models/144142
    - Clay: https://civitai.com/models/143769
    - LucasArts Game: https://civitai.com/models/151539/lucasarts-style-1990s-pc-adventure-games-sdxl-lora-dreambooth-trained?modelVersionId=169452
    - App Icons: https://civitai.com/models/149101?modelVersionId=166450
    - Voxel: https://civitai.com/models/118536
    - Watercolor: https://civitai.com/models/120789?modelVersionId=131382
    - Medieval Illustration: https://civitai.com/models/121290?modelVersionId=131944
    - Luigi Serafini: https://civitai.com/models/129910
    - Bad Quality: https://civitai.com/models/259627
    - HR Giger Rainbow: https://civitai.com/models/270990?modelVersionId=305459
    - Stop Motion Style: https://civitai.com/models/212442
    - Atari Style: https://civitai.com/models/468237/atari-2600-activision
    - Walking Dead Zombies: https://civitai.com/models/469054/the-walking-dead-zombies
    - InfernoFrames: https://civitai.com/models/203975/infernoframes-the-comics-from-beyond-the-grave
    - More LORAs (SDXL): https://civitai.com/user/Thaevilone/models
    - SDXL LORA comparison: https://www.reddit.com/r/StableDiffusion/comments/19fly6v/tested_fifteen_artistic_sdxl_loras_from_civitai/
    - Aardman: https://civitai.com/models/62212/aardman-animations-style
    - xlmoreart: https://civitai.com/models/124347/xlmoreart-full-xlreal-enhancer
  - SD 1.5
    - Artstyle Jackson Pollock: https://civitai.com/models/181433/artstyle-jackson-pollock
    - Real Mechanical Parts: https://civitai.com/models/64471?modelVersionId=85371
    - Cinnamon Bun: https://civitai.com/models/182311/cinnamon-bun-style-make-anything-sweet
    - Croissant: https://civitai.com/models/113365/croissant-style-croissantify-anything?modelVersionId=122454
    - Aardman: https://civitai.com/models/125317?modelVersionId=136883
    - B3L: https://civitai.com/models/167872/b3l
    - Vulcan: https://civitai.com/models/129132?modelVersionId=141479
    - ColorSplash: https://civitai.com/models/247981/colorsplash?modelVersionId=279794
    - Transformers Style: https://civitai.com/models/216460
    - Vector Illustration v2.0: https://civitai.com/models/60132/vector-illustration
- Tools
  - Krita SD Plugin: https://github.com/Acly/krita-ai-diffusion
  - SDNext (Automatic1111 fork): https://github.com/vladmandic/automatic
  - OneTrainer: https://github.com/Nerogar/OneTrainer
    - Training LORAs: https://www.reddit.com/r/StableDiffusion/comments/1cmxtnx/whats_your_preferred_method_to_train_sdxl_lora/
    - ![0018uf](https://github.com/mh-ka/ai-stuff/assets/52745439/fec433e9-15e6-42f3-80e6-80064f0ddfca)
  - DiffusionToolkit (AI Image Viewer): https://github.com/RupertAvery/DiffusionToolkit
  - Fooocus-inswapper (face swapping): https://github.com/machineminded/Fooocus-inswapper
  - Swifty Animator: https://tavurth.itch.io/swifty-animator
  - Style Aligned WebUI: https://github.com/cocktailpeanut/StyleAligned.pinokio
  - LCM Drawing App (local): https://github.com/kidintwo3/LCMDraw
  - Pallaidium (SD for Blender): https://github.com/tin2tin/Pallaidium
  - IC-Light: https://github.com/lllyasviel/IC-Light
  - Batch-tagging images
    - TagGUI: https://github.com/jhc13/taggui/releases
    - Image-Interrogator: https://github.com/DEVAIEXP/image-interrogator
  - ComfyUI:
    - IP Adapter: https://github.com/cubiq/ComfyUI_IPAdapter_plus
    - InstantID ComfyUI workflow: https://github.com/cubiq/ComfyUI_InstantID
    - SDXL Turbo workflow: https://comfyanonymous.github.io/ComfyUI_examples/sdturbo/
    - ComfyUI Deploy: https://github.com/BennyKok/comfyui-deploy
    - ComfyUI Manager: https://civitai.com/models/71980/comfyui-manager
    - Animation ControlNet guide: https://civitai.com/articles/3172/nature-powers-netflix-seasons-workflow-and-details
    - Refiner workflow: https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Experimental/sdxl-reencode/exp1.md
    - ELLA wrapper: https://github.com/kijai/ComfyUI-ELLA-wrapper
    - AniPortrait: https://github.com/chaojie/ComfyUI-AniPortrait
    - SUPIR Upscaler: https://github.com/kijai/ComfyUI-SUPIR
    - InstantID Style Composition workflow: https://civitai.com/models/423960
    - DynamicCrafterWrapper (for ToonCrafter): https://github.com/kijai/ComfyUI-DynamiCrafterWrapper
    - V-Express: https://github.com/AIFSH/ComfyUI_V-Express
    - MusePose: https://github.com/TMElyralab/Comfyui-MusePose
    - VLM Nodes: https://github.com/gokayfem/ComfyUI_VLM_nodes
    - PixArt Sigma Workflow: https://civitai.com/models/420163
    - Vid2Vid ComfyUI workflow: https://openart.ai/workflows/elephant_present_36/vid2vid_animatediff-hires-fix-face-detailer-hand-detailer-upscaler-mask-editor
    - AnimateDiff workflow: https://civitai.com/articles/2379/guide-comfyui-animatediff-guideworkflows-including-prompt-scheduling-an-inner-reflections-guide
    - Mega workflow: https://perilli.com/ai/comfyui/
    - LivePortrait: https://github.com/kijai/ComfyUI-LivePortraitKJ
    - AdvancedLivePortrait: https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait
  - Automatic1111 Extensions:
    - CADS (diversity): https://github.com/v0xie/sd-webui-cads
    - Forge (Speed and RAM optimizer): https://github.com/lllyasviel/stable-diffusion-webui-forge
    - LORA trainer: https://github.com/hako-mikan/sd-webui-traintrain
    - Nvidia TensorRT: https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT
    - Kohya HiRes Fix: https://github.com/wcde/sd-webui-kohya-hiresfix
    - Vector Studio: https://github.com/GeorgLegato/stable-diffusion-webui-vectorstudio
    - ControlNet (including InstantStyle): https://github.com/Mikubill/sd-webui-controlnet
      - Infos about InstantID: https://github.com/Mikubill/sd-webui-controlnet/discussions/2589 
    - ReActor: https://github.com/Gourieff/sd-webui-reactor
    - Inpaint Anything: https://github.com/Uminosachi/sd-webui-inpaint-anything
    - OneDiff: https://github.com/siliconflow/onediff/tree/main/onediff_sd_webui_extensions
    - Tabs: https://github.com/Haoming02/sd-webui-tabs-extension
    - Stylez: https://github.com/javsezlol1/Stylez
    - LobeTheme: https://github.com/lobehub/sd-webui-lobe-theme
    - De-oldify: https://github.com/SpenserCai/sd-webui-deoldify
  - Create textures with stable diffusion: https://stableprojectorz.com
- Misc
  - I Made Stable Diffusion XL Smarter: https://minimaxir.com/2023/08/stable-diffusion-xl-wrong/
  - Intel Arc and Microsoft support for stable diffusion: https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-and-Microsoft-Collaborate-to-Optimize-DirectML-for-Intel/post/1542055
  - DeciDiffusion v2.0: https://huggingface.co/Deci/DeciDiffusion-v2-0
  /LwT4AB8wEtKQilrjq2G8
  - FastSD CPU: https://github.com/rupeshs/fastsdcpu
  - Hints, lists and collections
    - SD Tutorials https://www.reddit.com/r/StableDiffusion/comments/yknrjt/list_of_sd_tutorials_resources/
    - Prompt Engineering Tools: https://www.reddit.com/r/StableDiffusion/comments/xcrm4d/useful_prompt_engineering_tools_and_resources/
    - SD Systems: https://www.reddit.com/r/StableDiffusion/comments/wqaizj/list_of_stable_diffusion_systems/
    - Dirty tricks: https://www.reddit.com/r/StableDiffusion/comments/1942m2y/what_are_your_dirty_tricks/
    - Difficult concepts: https://www.reddit.com/r/StableDiffusion/comments/1d40yyh/what_are_some_hard_or_difficult_concepts_you/
    - Must-Have Automatic1111 extensions: https://www.reddit.com/r/StableDiffusion/comments/18xj5mu/what_are_some_musthave_automatic1111_extensions/
    - More extension recommendations: https://www.reddit.com/r/StableDiffusion/comments/176jpuo/which_extensions_do_you_recommend_the_most/
    - Improve Image Quality: https://www.reddit.com/r/StableDiffusion/comments/1ck69az/make_it_good_options_in_stable_diffusion/
    - Vid2Vid Guide: https://civitai.com/articles/3194
    - Tutorials: https://www.patreon.com/jerrydavos
    - Public prompts: https://publicprompts.art/
    - More Prompts: https://prompthero.com/stable-diffusion-prompts
    - Art style prompts: https://github.com/scroobius-pip/arible_prompts/blob/main/prompts.arible
    - Upscaling Models: https://openmodeldb.info/
    - SD Models: https://rentry.org/sdmodels
    - Image Collection: https://civitai.com/collections/15937?sort=Most+Collected
    - ELLA tutorial: https://sandner.art/ella-leveraging-llms-for-enhanced-semantic-alignment-in-sd-15/
    - LORA training tutorial: https://rentry.org/PlumLora

<sup><sub>[back to top](#toc)</sub></sup>

### Flux
- LORAs
  - CCTV: https://civitai.com/models/689331/convenience-store-cctv-or-flux1-dev
  - PS1/PS2: https://civitai.com/models/638052/ps1-ps2-old-3d-game-style-flux
  - Phlux: https://civitai.com/models/672963/phlux-photorealism-with-style-incredible-texture-and-lighting
  - Naoki Urasawa: https://civitai.com/models/690155/naoki-urasawa-manga-style-flux-lora?modelVersionId=772410
  - Icon Maker: https://civitai.com/models/722531
  - Oil Painting: https://civitai.com/models/723141
  - Pixel Art: https://civitai.com/models/672328
  - 1999 Digital Camera Style: https://civitai.com/models/724495
  - Dashed Lines: https://civitai.com/models/724476
  - Amateur Photo: https://civitai.com/models/652699
- Misc
  - LORA search engine: https://www.fluxforge.app/
  - JoyCaption: https://huggingface.co/Wi-zz/joy-caption-pre-alpha
  - Flux Gym: https://github.com/cocktailpeanut/fluxgym
  - Flux Latent Upscaler: https://github.com/rickrender/FluxLatentUpscaler

<sup><sub>[back to top](#toc)</sub></sup>

### Animation and Video etc
- AnimateDiff
  - Prompt Travel Video2Video Tutorial: https://stable-diffusion-art.com/animatediff-prompt-travel-video2video/
  - AnimateDiff CLI Prompt Travel: https://github.com/s9roll7/animatediff-cli-prompt-travel
  - ComfyUI Guide: https://civitai.com/articles/2379
  - ComfyUI Node Setup: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
  - AnimateDiff-Lightning: https://huggingface.co/ByteDance/AnimateDiff-Lightning
  - AnimateDiff Video Tutorial: https://www.youtube.com/watch?v=Gz9pT2CwdoI
- ToonCrafter: https://github.com/ToonCrafter/ToonCrafter
- Video face swap: https://github.com/s0md3v/roop
- Removal of Backgrounds: https://huggingface.co/briaai/RMBG-1.4
- Pixart-alpha/-delta: https://github.com/PixArt-alpha/PixArt-alpha
- Pixart-sigma (release soon): https://github.com/PixArt-alpha/PixArt-sigma
- How to use Midjourney to design a brand identity: https://www.completepython.com/how-i-used-midjourney-to-design-a-brand-identity/
- Alibaba VGen (video generation projects): https://github.com/ali-vilab/VGen
- ClioBot: https://github.com/herval/cliobot
- InstantMesh (img -> 3D Mesh): https://github.com/TencentARC/InstantMesh
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- BIRD (Image restoration): https://github.com/hamadichihaoui/BIRD
- Blog around image detection techniques: https://blog.roboflow.com/
- WildGaussians: https://github.com/jkulhanek/wild-gaussians/
- CogVideoX-5b: https://huggingface.co/THUDM/CogVideoX-5b

<sup><sub>[back to top](#toc)</sub></sup>


## Reports and Articles and Sources

### Reports
- Kaggle AI Report 2023: https://www.kaggle.com/AI-Report-2023
- State of AI Report 2023: https://docs.google.com/presentation/d/156WpBF_rGvf4Ecg19oM1fyR51g4FAmHV3Zs0WLukrLQ/preview?slide=id.g24daeb7f4f0_0_3445
- State of Open Source AI Book: https://book.premai.io/state-of-open-source-ai/index.html
- Foundation Model Transparency Index: https://crfm.stanford.edu/fmti/May-2024/index.html
- AI Timeline: https://nhlocal.github.io/AiTimeline/#2024

<sup><sub>[back to top](#toc)</sub></sup>

### Articles
- Geospatial for Earth Observations (NASA, IBM): https://huggingface.co/ibm-nasa-geospatial
- Graphcast for Weather Forecast (Google): https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/
- RT-2 for Robot Instructions (Google): https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/
- DragGAN for Image Manipulation (Google): https://vcai.mpi-inf.mpg.de/projects/DragGAN/
- Replicate Guide on upscaling images: https://replicate.com/guides/upscaling-images
- Github Copilot's impact on productivity: https://cacm.acm.org/research/measuring-github-copilots-impact-on-productivity/
- AI security testing github projects: https://github.com/AnthenaMatrix
- GenAI overview with links and details: https://medium.com/@maximilian.vogel/5000x-generative-ai-intro-overview-models-prompts-technology-tools-comparisons-the-best-a4af95874e94
- GenAI link list: https://github.com/steven2358/awesome-generative-ai
- ChatGPT list of lists: https://medium.com/@maximilian.vogel/the-chatgpt-list-of-lists-a-collection-of-1500-useful-mind-blowing-and-strange-use-cases-8b14c35eb
- AGI Predictions: https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf
- Robocasa (large-scale simulation framework for robot training): https://github.com/robocasa/robocasa

<sup><sub>[back to top](#toc)</sub></sup>

### News sources
- Newsletter
  - https://substack.com/@aibrews
  - https://lifearchitect.substack.com/
  - https://simonwillison.net/
  - https://diffusiondigest.beehiiv.com/
- Websites
  - https://the-decoder.de/
  - http://futuretools.io/
  - https://lifearchitect.ai/
- Youtube Channels
  - LLM News: https://www.youtube.com/@DrAlanDThompson
  - AI Tools News
    - https://www.youtube.com/@mreflow
    - https://www.youtube.com/@MattVidPro
  - Stable Diffusion Workflows:
    - https://www.youtube.com/@NerdyRodent
    - https://www.youtube.com/@sebastiankamph

<sup><sub>[back to top](#toc)</sub></sup>
