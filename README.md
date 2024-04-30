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
  * [Image and video generation](#image-and-video-generation)
    + [Stable diffusion](#stable-diffusion)
    + [AnimateDiff](#animatediff)
    + [Other](#other-1)
  * [Misc](#misc)
    + [Reports](#reports)
    + [Other](#other-2)
    + [News sources](#news-sources)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


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
- Tokenization: https://www.youtube.com/watch?v=zduSFxRajkE
- Prompt Injection and Jailbreaking: https://simonwillison.net/2024/Mar/5/prompt-injection-jailbreaking/
- Insights to model file formats: https://vickiboykis.com/2024/02/28/gguf-the-long-way-around/
- Prompt Library: https://www.moreusefulthings.com/prompts
- Mixture of Experts explained (dense vs. sparse models): https://alexandrabarr.beehiiv.com/p/mixture-of-experts

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
- MaLA: https://huggingface.co/MaLA-LM/mala-500
- Starcoder 2: https://github.com/bigcode-project/starcoder2
- DBRX Base and Instruct: https://github.com/databricks/dbrx
- Cohere Command-R: https://huggingface.co/CohereForAI/c4ai-command-r-v01
- Small sized models
  - Microsoft phi-2 2.7B: https://huggingface.co/microsoft/phi-2
  - Replit Code v1.5 3B for coding: https://huggingface.co/replit/replit-code-v1_5-3b
  - RWKV-5 Eagle 7B: https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers
  - Google
    - Gemma 2B and 7B: https://huggingface.co/blog/gemma
    - CodeGemma: https://www.kaggle.com/models/google/codegemma
  - Moondream (tiny vision model): https://github.com/vikhyat/moondream
  - Yi-9B: https://huggingface.co/01-ai/Yi-9B
- Multimodal models
  - LLaVA: https://github.com/haotian-liu/LLaVA
    - First Impressions with LLaVA 1.5: https://blog.roboflow.com/first-impressions-with-llava-1-5/
    - LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/
  - Apple Ferret: https://github.com/apple/ml-ferret
  - Alibaba Cloud Qwen-VL: https://qwenlm.github.io/blog/qwen-vl/
  - DeepSeek-VL: https://github.com/deepseek-ai/DeepSeek-VL
- Ollama library: https://ollama.com/library

### Tools around using LLMs
- Local LLM hosting
  - vLLM (local python library for running models): https://docs.vllm.ai/en/latest/getting_started/quickstart.html
  - LMStudio (local model hosting): https://lmstudio.ai/
  - Jan app (local model hosting): https://jan.ai/
  - Leeroo Orchestrator (choose model depending on task): https://github.com/leeroo-ai/leeroo_orchestrator
  - FreedomGPT (local model hosting): https://www.freedomgpt.com/
  - ExLlama2 (optimized local inference): https://github.com/turboderp/exllamav2
  - pinokio (Simple app for running many AI tools locally): https://pinokio.computer/
  - GPT4All (local LLM hosting): https://gpt4all.io/index.html
  - h2oGPT (Private Q&A with your own documents): https://github.com/h2oai/h2ogpt
  - chatd (chat with your own documents): https://github.com/BruceMacD/chatd
  - Nextcloud AI assistant: https://docs.nextcloud.com/server/latest/admin_manual/ai/index.html (https://github.com/nextcloud/all-in-one/tree/main)
  - Opera One Developer (browser with LLMs): https://blogs.opera.com/news/2024/04/ai-feature-drops-local-llms/
- Local coding assistance
  - Open Interpreter (local model hosting with code execution and file input): https://github.com/KillianLucas/open-interpreter
  - Tabby (Self-hosted coding assistant): https://tabby.tabbyml.com/docs/getting-started
  - Sweep (AI junior developer): https://github.com/sweepai/sweep
  - GPT-engineer (let GPT write and run code): https://github.com/gpt-engineer-org/gpt-engineer
  - OSS alternative to Devin AI Software Engineer: https://github.com/stitionai/devika
- More coding assistance
  - Aider (coding): https://github.com/paul-gauthier/aider
  - Cursor (GPT-API-based code editor app): https://cursor.sh/
  - Plandex (GPT-API-based coding agent): https://github.com/plandex-ai/plandex
  - SWE-agent (another GPT-API-based coding agent): https://github.com/princeton-nlp/SWE-agent
  - OpenUI (another GPT-API-based coding agent for web components): https://github.com/wandb/openui
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

### AI Developer Topics
- How to design an agent for production: https://blog.langchain.dev/how-to-design-an-agent-for-production/
- LLM App Stack: https://github.com/a16z-infra/llm-app-stack
- Fine-tune NLP models: https://towardsdatascience.com/domain-adaption-fine-tune-pre-trained-nlp-models-a06659ca6668
- Document-Oriented Agents: https://towardsdatascience.com/document-oriented-agents-a-journey-with-vector-databases-llms-langchain-fastapi-and-docker-be0efcd229f4
- Open questions for AI Engineering: https://simonwillison.net/2023/Oct/17/open-questions/#slides-start
- List of AI developer tools: https://github.com/sidhq/YC-alum-ai-tools
- Prem AI infrastructure tooling: https://github.com/premAI-io/prem-app
- TensorBoard for TensorFlow visualization: https://www.tensorflow.org/tensorboard
- RAGs ("Build ChatGPT over your data"): https://github.com/run-llama/rags
- Docker GenAI Stack: https://github.com/docker/genai-stack
- How to use consumer hardware to train 70b LLM: https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Langchain Templates (templates for building AI apps): https://github.com/langchain-ai/langchain/tree/master/templates
- Dify (local LLM app development platform): https://github.com/langgenius/dify
- How to construct domain-specific LLM evaluation systems: https://hamel.dev/blog/posts/evals/
- How to get great AI code completions (technical insights on code completion): https://sourcegraph.com/blog/the-lifecycle-of-a-code-ai-completion
- Datasets: https://archive.ics.uci.edu/

### Evaluation
- Big Code Models Leaderboard: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
- Chatbot Arena: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
- Zephyr 7B infos on reddit: https://www.reddit.com/r/LocalLLaMA/comments/17hjgdg/zephyr_7b_beta_a_new_mistral_finetune_is_out/


## Audio

### Voicecraft
- Repo: https://github.com/jasonppy/VoiceCraft
- Huggingface: https://huggingface.co/pyp1/VoiceCraft/tree/main
- Docker version: https://github.com/pselvana/VoiceCrafter
- API version: https://github.com/lukaszliniewicz/VoiceCraft_API

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

### txt2speech and txt2audio
- XTTS v1: https://huggingface.co/coqui/XTTS-v1
- AudioLDM2: https://github.com/haoheliu/audioldm2
- Bark: https://github.com/suno-ai/bark
- Tracker page for open access text2speech models: https://github.com/Vaibhavs10/open-tts-tracker
- MetaVoice: https://github.com/metavoiceio/metavoice-src
- Pheme TTS framework: https://github.com/PolyAI-LDN/pheme
- OpenAI TTS: https://platform.openai.com/docs/guides/text-to-speech
- TTS Arena Leaderboard: https://huggingface.co/spaces/TTS-AGI/TTS-Arena
- Parler TTS: https://github.com/huggingface/parler-tts

### Music production
- LANDR mastering plugin: https://www.gearnews.de/landr-mastering-plugin/
- Drumloop.ai: https://www.gearnews.de/drumloop-ai-baut-euch-automatisch-beats-und-drumloops-durch-ki/

### Other
- LAION AI Voice Assistant BUD-E: https://github.com/LAION-AI/natural_voice_assistant
- AI Language Tutor:
  - https://www.univerbal.app/
  - https://yourteacher.ai/
- Speech Note Offline STT, TTS and Machine Translation: https://github.com/mkiol/dsnote

## Image and video generation

### Stable diffusion
- Models
  - Stable Video Diffusion 1.1: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
  - Stable Cascade: https://stability.ai/news/introducing-stable-cascade
  - DreamShaper XL Turbo v2: https://civitai.com/models/112902/dreamshaper-xl
  - Amiga Style: https://civitai.com/models/297955/amiga-style
  - Anime Model: https://huggingface.co/XpucT/Anime
  - GhostMix: https://civitai.com/models/36520/ghostmix
  - ReV Animated: https://civitai.com/models/7371/rev-animated
  - Animagine XL 3.0: https://huggingface.co/cagliostrolab/animagine-xl-3.0
  - Cheyenne Comic Model: https://civitai.com/models/198051/cheyenne?modelVersionId=318969
  - EpicPhotoGasm: https://civitai.com/models/132632/epicphotogasm
  - Nebul.Redmond (SDXL Finetune): https://civitai.com/models/269988
  - Pixel Art Diffusion XL: https://civitai.com/models/277680/pixel-art-diffusion-xl
  - Juggernaut XL v7: https://civitai.com/models/133005?modelVersionId=240840
  - Juggernaut XL v9: https://huggingface.co/RunDiffusion/Juggernaut-XL-v9
  - ThinkDiffusionXL: https://civitai.com/models/169868/thinkdiffusionxl
  - Realism Engine SDXL: https://civitai.com/models/152525
  - LEOSAM AIArt: https://civitai.com/models/219791?modelVersionId=247832
  - Haveall: https://civitai.com/models/213692
  - Coloring Book: https://civitai.com/models/5153/coloring-book
  - Vivid Watercolors: https://civitai.com/models/4998/vivid-watercolors
  - RZ Vice Embedding: https://civitai.com/models/4920/rz-vice-21
  - Ghibli Diffusion: https://civitai.com/models/1066/ghibli-diffusion
  - Cool Japan Diffusion: https://huggingface.co/aipicasso/cool-japan-diffusion-2-1-1
  - UnstableInkDream: https://civitai.com/models/1540/unstableinkdream
  - Vintage Airbrushed: https://civitai.com/models/4291
  - Berlin Graffiti: https://huggingface.co/bakebrain/bergraffi-berlin-graffiti
  - Macro Diffusion: https://civitai.com/models/3863/macro-diffusion
  - Favorite Models: https://www.reddit.com/r/StableDiffusion/comments/19fdm0c/whats_your_favorite_model_currently/
- LORAs
  - Bad Quality: https://civitai.com/models/259627
  - HR Giger Rainbow: https://civitai.com/models/270990?modelVersionId=305459
  - Remix XL: https://huggingface.co/Stkzzzz222/remixXL
  - iPhone Mirror Selfie: https://civitai.com/models/256058/iphone-mirror-selfie-or-sdxl
  - ColorSplash: https://civitai.com/models/247981/colorsplash?modelVersionId=279794
  - Transformers Style: https://civitai.com/models/216460
  - Stop Motion Style: https://civitai.com/models/212442
  - Aether Aqua: https://civitai.com/models/210754/aether-aqua-lora-for-sdxl
  - Claymation Style: https://civitai.com/models/208168/claymate-stop-motion-claymation-style-for-sdxl
  - SDXL LORA comparison: https://www.reddit.com/r/StableDiffusion/comments/19fly6v/tested_fifteen_artistic_sdxl_loras_from_civitai/
- Tools
  - Krita SD Plugin: https://github.com/Acly/krita-ai-diffusion
  - SDNext (Automatic1111 fork): https://github.com/vladmandic/automatic
  - OneTrainer: https://github.com/Nerogar/OneTrainer
  - DiffusionToolkit (AI Image Viewer): https://github.com/RupertAvery/DiffusionToolkit
  - Fooocus-inswapper (face swapping): https://github.com/machineminded/Fooocus-inswapper
  - Swifty Animator: https://tavurth.itch.io/swifty-animator
  - Style Aligned WebUI: https://github.com/cocktailpeanut/StyleAligned.pinokio?tab=readme-ov-file
  - LCM Drawing App (local): https://github.com/kidintwo3/LCMDraw
  - Pallaidium (SD for Blender): https://github.com/tin2tin/Pallaidium
  - ComfyUI:
    - InstantID ComfyUI workflow: https://github.com/cubiq/ComfyUI_InstantID
    - SDXL Turbo workflow: https://comfyanonymous.github.io/ComfyUI_examples/sdturbo/
    - ComfyUI Deploy: https://github.com/BennyKok/comfyui-deploy
    - ComfyUI Manager: https://civitai.com/models/71980/comfyui-manager
    - Animation ControlNet guide: https://civitai.com/articles/3172/nature-powers-netflix-seasons-workflow-and-details
    - Refiner workflow: https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Experimental/sdxl-reencode/exp1.md
    - ELLA wrapper: https://github.com/kijai/ComfyUI-ELLA-wrapper
    - AniPortrait: https://github.com/chaojie/ComfyUI-AniPortrait
  - Automatic1111 Extensions:
    - CADS (diversity): https://github.com/v0xie/sd-webui-cads
    - Forge (Speed and RAM optimizer): https://github.com/lllyasviel/stable-diffusion-webui-forge
    - LORA trainer: https://github.com/hako-mikan/sd-webui-traintrain
    - Nvidia TensorRT: https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT
    - Kohya HiRes Fix: https://github.com/wcde/sd-webui-kohya-hiresfix
    - Vector Studio: https://github.com/GeorgLegato/stable-diffusion-webui-vectorstudio
- Misc
  - OneDiff (Speed optimisied SDXL): https://github.com/siliconflow/onediff
  - I Made Stable Diffusion XL Smarter: https://minimaxir.com/2023/08/stable-diffusion-xl-wrong/
  - Intel Arc and Microsoft support for stable diffusion: https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-and-Microsoft-Collaborate-to-Optimize-DirectML-for-Intel/post/1542055
  - DeciDiffusion v2.0: https://huggingface.co/Deci/DeciDiffusion-v2-0
  - Tutorials: https://www.patreon.com/jerrydavos
  - Vid2Vid ComfyUI workflow: https://openart.ai/workflows/elephant_present_36/vid2vid_animatediff-hires-fix-face-detailer-hand-detailer-upscaler-mask-editor/LwT4AB8wEtKQilrjq2G8
  - Dirty tricks: https://www.reddit.com/r/StableDiffusion/comments/1942m2y/what_are_your_dirty_tricks/
  - Must-Have Automatic1111 extensions: https://www.reddit.com/r/StableDiffusion/comments/18xj5mu/what_are_some_musthave_automatic1111_extensions/
  - FastSD CPU: https://github.com/rupeshs/fastsdcpu
  - Vid2Vid Guide: https://civitai.com/articles/3194
  - Art style prompts: https://github.com/scroobius-pip/arible_prompts/blob/main/prompts.arible
  - ELLA tutorial: https://sandner.art/ella-leveraging-llms-for-enhanced-semantic-alignment-in-sd-15/
  - Lists:
    - https://www.reddit.com/r/StableDiffusion/comments/yknrjt/list_of_sd_tutorials_resources/
    - https://www.reddit.com/r/StableDiffusion/comments/xcrm4d/useful_prompt_engineering_tools_and_resources/
    - https://www.reddit.com/r/StableDiffusion/comments/wqaizj/list_of_stable_diffusion_systems/
    - https://diffusiondb.com/
    - https://github.com/steven2358/awesome-generative-ai
  - Prompts: https://publicprompts.art/
  - More Prompts: https://prompthero.com/stable-diffusion-prompts
  - Upscaling Models: https://openmodeldb.info/
  - Models: https://rentry.org/sdmodels

### AnimateDiff
- Prompt Travel Video2Video Tutorial: https://stable-diffusion-art.com/animatediff-prompt-travel-video2video/
- AnimateDiff CLI Prompt Travel: https://github.com/s9roll7/animatediff-cli-prompt-travel
- ComfyUI Guide: https://civitai.com/articles/2379
- ComfyUI Node Setup: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
- AnimateDiff-Lightning: https://huggingface.co/ByteDance/AnimateDiff-Lightning
- AnimateDiff Video Tutorial: https://www.youtube.com/watch?v=Gz9pT2CwdoI

### Other
- Video face swap: https://github.com/s0md3v/roop
- Removal of Backgrounds: https://huggingface.co/briaai/RMBG-1.4
- Pixart-alpha/-delta: https://github.com/PixArt-alpha/PixArt-alpha?tab=readme-ov-file#breaking-news-
- Pixart-sigma (release soon): https://github.com/PixArt-alpha/PixArt-sigma
- How to use Midjourney to design a brand identity: https://www.completepython.com/how-i-used-midjourney-to-design-a-brand-identity/
- Alibaba VGen (video generation projects): https://github.com/ali-vilab/VGen
- ClioBot: https://github.com/herval/cliobot


## Misc

### Reports
- Kaggle AI Report 2023: https://www.kaggle.com/AI-Report-2023
- State of AI Report 2023: https://docs.google.com/presentation/d/156WpBF_rGvf4Ecg19oM1fyR51g4FAmHV3Zs0WLukrLQ/preview?slide=id.g24daeb7f4f0_0_3445
- State of Open Source AI Book: https://book.premai.io/state-of-open-source-ai/index.html

### Other
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
- Github Copilot's impact on productivity: https://cacm.acm.org/research/measuring-github-copilots-impact-on-productivity/
- AI security testing github projects: https://github.com/AnthenaMatrix
- GenAI overview with links and details: https://medium.com/@maximilian.vogel/5000x-generative-ai-intro-overview-models-prompts-technology-tools-comparisons-the-best-a4af95874e94
- ChatGPT list of lists: https://medium.com/@maximilian.vogel/the-chatgpt-list-of-lists-a-collection-of-1500-useful-mind-blowing-and-strange-use-cases-8b14c35eb

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
