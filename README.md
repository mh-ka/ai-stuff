# AI Stuff


## LLMs

### Basics
- What are LLMs?: https://ig.ft.com/generative-ai/
- What are Embeddings: https://simonwillison.net/2023/Oct/23/embeddings/
- Visualization of how a LLM does work: https://bbycroft.net/llm
- How AI works (high level explanation): https://every.to/p/how-ai-works
- How to use AI to do stuff: https://www.oneusefulthing.org/p/how-to-use-ai-to-do-stuff-an-opinionated
- Catching up on the weird world of LLMs: https://simonwillison.net/2023/Aug/3/weird-world-of-llms/
- Prompt Engineering Guide: https://github.com/dair-ai/Prompt-Engineering-Guide

### Models
- Meta's Llama
  - CodeLLama: https://github.com/facebookresearch/codellama
  - slowllama: with offloading to SSD/Memory: https://github.com/okuvshynov/slowllama
- Mistral's Finetunes & Mixtral
  - Finetune Mistral on your own data: https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb
  - Mixtral: https://huggingface.co/blog/mixtral
- Phind-CodeLlama-34B v2: https://huggingface.co/Phind/Phind-CodeLlama-34B-v2
- Doctor Dignity: https://github.com/llSourcell/Doctor-Dignity
- BigTranslate: https://github.com/ZNLP/BigTranslate
- DeepSeek-LLM 67B: https://github.com/deepseek-ai/DeepSeek-LLM
- Alibaba's Qwen 1.5: https://qwenlm.github.io/blog/qwen1.5/
- Small sized models
  - Microsoft phi-2 2.7B: https://huggingface.co/microsoft/phi-2
  - Replit Code v1.5 3B for coding: https://huggingface.co/replit/replit-code-v1_5-3b
  - RWKV-5 Eagle 7B: https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers
- Multimodal models
  - LLaVA: https://github.com/haotian-liu/LLaVA
    - First Impressions with LLaVA 1.5: https://blog.roboflow.com/first-impressions-with-llava-1-5/
    - LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/
  - Apple Ferret: https://github.com/apple/ml-ferret
  - Alibaba Cloud Qwen-VL: https://qwenlm.github.io/blog/qwen-vl/

### Tools around using LLMs
- vLLM (local python library for running models): https://docs.vllm.ai/en/latest/getting_started/quickstart.html
- LMStudio (local model hosting): https://lmstudio.ai/
- Langchain Templates (templates for building AI apps): https://github.com/langchain-ai/langchain/tree/master/templates
- FreedomGPT (local model hosting): https://www.freedomgpt.com/
- ExLlama2 (optimized local inference): https://github.com/turboderp/exllamav2
- Phind (AI based search engine): https://www.phind.com/search?home=true
- pinokio (Simple app for running many AI tools locally): https://pinokio.computer/
- Tabby (Self-hosted coding assistant): https://tabby.tabbyml.com/docs/getting-started
- Open Interpreter (Use LLM APIs to run local code interpreter): https://github.com/KillianLucas/open-interpreter
- Scalene (High Performance Python Profiler): https://github.com/plasma-umass/scalene
- Cursor (GPT-API-based code editor app): https://cursor.sh/
- Lida (visualization and infographics generated with LLMs): https://github.com/microsoft/lida
- OpenCopilot (LLM integration with provided Swagger APIs): https://github.com/openchatai/OpenCopilot
- LogAI (log analytics): https://github.com/salesforce/logai
- Sweep (AI junior developer): https://github.com/sweepai/sweep
- Jupyter AI (jupyter notebook AI assistance): https://github.com/jupyterlab/jupyter-ai
- h2oGPT (Private Q&A with your own documents): https://github.com/h2oai/h2ogpt
- GPT-engineer (let GPT write and run code): https://github.com/gpt-engineer-org/gpt-engineer
- GPT4All (local LLM hosting): https://gpt4all.io/index.html
- chatd (chat with your own documents): https://github.com/BruceMacD/chatd

### AI Engineer / Developer Topics
- How to design an agent for production: https://blog.langchain.dev/how-to-design-an-agent-for-production/
- LLM App Stack: https://a16z.com/emerging-architectures-for-llm-applications/
- Fine-tune NLP models: https://towardsdatascience.com/domain-adaption-fine-tune-pre-trained-nlp-models-a06659ca6668
- Document-Oriented Agents: https://towardsdatascience.com/document-oriented-agents-a-journey-with-vector-databases-llms-langchain-fastapi-and-docker-be0efcd229f4
- Open questions for AI Engineering: https://simonwillison.net/2023/Oct/17/open-questions/#slides-start
- List of AI developer tools: https://github.com/sidhq/YC-alum-ai-tools
- Prem AI infrastructure tooling: https://github.com/premAI-io/prem-app
- TensorBoard for TensorFlow visualization: https://www.tensorflow.org/tensorboard
- RAGs ("Build ChatGPT over your data"): https://github.com/run-llama/rags
- Docker GenAI Stack: https://github.com/docker/genai-stack

### Evaluation
- Big Code Models Leaderboard: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
- Chatbot Arena: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
- Zephyr 7B infos on reddit: https://www.reddit.com/r/LocalLLaMA/comments/17hjgdg/zephyr_7b_beta_a_new_mistral_finetune_is_out/


## Audio

### Meta
- MusicGen: https://huggingface.co/facebook/musicgen-stereo-melody-large
  - Trying out MusicGen: https://til.simonwillison.net/machinelearning/musicgen
- SeamlessStreaming: https://huggingface.co/facebook/seamless-streaming
  - Seamless Framework: https://github.com/facebookresearch/seamless_communication
- MAGNeT (txt-2-music/-audio): https://analyticsindiamag.com/meta-launches-magnet-an-open-source-text-to-audio-model/
  - Hugging Face page with models: https://huggingface.co/collections/facebook/magnet-659ef0ceb62804e6f41d1466

### speech2txt
- OpenAI's Whisper: https://github.com/openai/whisper
  - Distil-Whisper: https://github.com/huggingface/distil-whisper/issues/4
  - Insanely fast whisper: https://github.com/Vaibhavs10/insanely-fast-whisper
  - WhisperKit for Apple devices: https://www.takeargmax.com/blog/whisperkit
- Nvidia's Canary (with translation): https://nvidia.github.io/NeMo/blogs/2024/2024-02-canary/

### txt2speech / txt2audio
- XTTS v1: https://huggingface.co/coqui/XTTS-v1
- AudioLDM2: https://github.com/haoheliu/audioldm2
- Bark: https://github.com/suno-ai/bark
- Tracker page for open access text2speech models: https://github.com/Vaibhavs10/open-tts-tracker
- MetaVoice: https://github.com/metavoiceio/metavoice-src
- Pheme TTS framework: https://github.com/PolyAI-LDN/pheme

### Music production
- LANDR mastering plugin: https://www.gearnews.de/landr-mastering-plugin/
- Drumloop.ai: https://www.gearnews.de/drumloop-ai-baut-euch-automatisch-beats-und-drumloops-durch-ki/

### Other
- LAION AI Voice Assistant BUD-E: https://github.com/LAION-AI/natural_voice_assistant


## Image/Video generation

### Stable diffusion
- I Made Stable Diffusion XL Smarter: https://minimaxir.com/2023/08/stable-diffusion-xl-wrong/
- Intel Arc and Microsoft support for stable diffusion: https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-and-Microsoft-Collaborate-to-Optimize-DirectML-for-Intel/post/1542055
- Krita SD Plugin: https://github.com/Acly/krita-ai-diffusion
- Stable Video Diffusion 1.1: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
- Stable Cascade: https://stability.ai/news/introducing-stable-cascade

### AnimateDiff
- Prompt Travel Video2Video: https://stable-diffusion-art.com/animatediff-prompt-travel-video2video/
- AnimateDiff CLI Prompt Travel: https://github.com/s9roll7/animatediff-cli-prompt-travel
- ComfyUI Guide: https://civitai.com/articles/2379
- ComfyUI Node Setup: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved

### Other
- Video face swap: https://github.com/s0md3v/roop
- Removal of Backgrounds: https://huggingface.co/briaai/RMBG-1.4
- Pixart-alpha/-delta: https://github.com/PixArt-alpha/PixArt-alpha?tab=readme-ov-file#breaking-news-


## Misc

### Reports
- Kaggle AI Report 2023: https://www.kaggle.com/AI-Report-2023
- State of AI Report 2023: https://docs.google.com/presentation/d/156WpBF_rGvf4Ecg19oM1fyR51g4FAmHV3Zs0WLukrLQ/preview?slide=id.g24daeb7f4f0_0_3445
- State of Open Source AI Book: https://book.premai.io/state-of-open-source-ai/index.html

### Research and Releases
- LLMs and political biases: https://www.technologyreview.com/2023/08/07/1077324/ai-language-models-are-rife-with-political-biases
- Using LLM to create a audio storyline: https://github.com/Audio-AGI/WavJourney
- Geospatial for Earth Observations (NASA, IBM): https://huggingface.co/ibm-nasa-geospatial
- Graphcast for Weather Forecast (Google): https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/
- RT-2 for Robot Instructions (Google): https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/
- DragGAN for Image Manipulation (Google): https://vcai.mpi-inf.mpg.de/projects/DragGAN/
- Pruning Approach for LLMs (Meta, Bosch): https://github.com/locuslab/wanda
- LagLlama (time series forecasting): https://github.com/time-series-foundation-models/lag-llama
- Mistral Guide on basic RAG: https://docs.mistral.ai/guides/basic-RAG/
- Replicate Guide on upscaling images: https://replicate.com/guides/upscaling-images

### News sources
- Newsletter
  - https://substack.com/@aibrews
  - https://lifearchitect.substack.com/
  - https://simonwillison.net/
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
