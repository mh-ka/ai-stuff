# Video / Image / 3D Generation

## Image Generation Models (besides Stable Diffusion and Flux)
- PixArt
  - [alpha/-delta](https://github.com/PixArt-alpha/PixArt-alpha)
  - [sigma](https://github.com/PixArt-alpha/PixArt-sigma)
  - [TinyBreaker (based on PixArt)](https://civitai.com/models/1213728/tinybreaker)
- Kolors
  - [Model](https://huggingface.co/Kwai-Kolors/Kolors )
  - [Workflow](https://civitai.com/models/1078853/midjourney-is-kolorflux)
- [List of popular txt2img generative models](https://github.com/vladmandic/automatic/wiki/Models#)
- [ComfyUI workflows for several diffusion models](https://github.com/city96/ComfyUI_ExtraModels)
- [DiffSensei (manga panels generator)](https://github.com/jianzongwu/DiffSensei)
- [Nvidia's Sana](https://github.com/NVlabs/Sana)
- [Lumina Image 2.0](https://civitai.com/models/1222266?modelVersionId=1377095)
- HiDream
  - [Model](https://github.com/HiDream-ai/HiDream-I1)
  - [Editing Model](https://github.com/HiDream-ai/HiDream-E1)
  - LORAs
    - [Coloring Book](https://civitai.com/models/1518899/coloring-book-hidream)
- [Marigold (Depth Estimation)](https://github.com/prs-eth/Marigold)
- [Quillworks 2.0](https://tensor.art/models/877299729996755011/Quillworks2.0-Experimental-2.0-Experimental)


## Image Manipulation with AI

### Removal of Backgrounds
- [RMBG](https://huggingface.co/briaai/RMBG-1.4)
- [InSPyReNet](https://github.com/plemeri/InSPyReNet)
- [BiRefNet: https://github.com/ZhengPeng7/BiRefNet (ComfyUI](https://github.com/1038lab/ComfyUI-RMBG))
- [BEN: https://huggingface.co/PramaLLC/BEN (ComfyUI](https://github.com/DoctorDiffusion/ComfyUI-BEN))
- [BEN2](https://huggingface.co/PramaLLC/BEN2)
- [Leaderboard](https://huggingface.co/spaces/bgsys/background-removal-arena)

### Face Swapping
- [Roop Unleashed](https://codeberg.org/rcthans/roop-unleashednew)
- [Roop Floyd](https://codeberg.org/Cognibuild/ROOP-FLOYD)
- [Rope-next](https://github.com/lodgecku/Rope-next)
- [VisoMaster](https://github.com/visomaster/VisoMaster)
  - [Fork with better GPU support](https://github.com/loscrossos/core_visomaster)
- FaceFusion (3.0.0)
  - [Github](https://github.com/facefusion/facefusion)
  - [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1fpbm3p/facefusion_300_has_finally_launched/)
- [ACE++](https://www.patreon.com/posts/face-swapping-121224741)
- [InfiniteYou (face to photo)](https://github.com/bytedance/InfiniteYou)
- [Nexface](https://github.com/ExoFi-Labs/Nexface/)
- [Reactor workflow](https://www.runcomfy.com/comfyui-workflows/comfyui-reactor-face-swap-professional-ai-face-animation)
- [Reactor node](https://github.com/Gourieff/ComfyUI-ReActor)
- Human Animation

### Restoration / colorization / pixelation / detailer
- [BIRD (Image restoration)](https://github.com/hamadichihaoui/BIRD)
- [Colorize Workflow](https://civitai.com/articles/15221)
- [Detail Daemon (adding details)](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
- [Detailer Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- Pixelation
  - https://github.com/filipemeneses/comfy_pixelization
  - https://github.com/WuZongWei6/Pixelization
- [Latent Bridge Matching (image relighting, restoration etc.)](https://github.com/gojasper/LBM)

### Upscaling
- [Siax](https://civitai.com/models/147641/nmkd-siax-cx)
- Ultrasharp
  - https://civitai.com/models/116225/4x-ultrasharp
  - https://huggingface.co/Kim2091/UltraSharpV2
- [Superscale](https://civitai.com/models/141491/4x-nmkd-superscale)
- [4x-IllustrationJaNai-V1-DAT2](https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2)
- [Upscayl](https://github.com/upscayl/upscayl)
- [Chain of Zoom](https://github.com/bryanswkim/Chain-of-Zoom)
- [ComfyUI_Steudio](https://github.com/Steudio/ComfyUI_Steudio)
<blockquote>
In addition to the bypassing the FluxKontextImageScale node and using reasonable image sizes, I usually pass the final image through the STEUDIO tiled upscale workflow https://github.com/Steudio/ComfyUI_Steudio , setting end_percent to 0.95 to preserve the identity, or lower when I don't care about the identity too much. The scale factor can be set to 1 if don't want to wait for long upscales, and tile size can be set to 1408, which seems to be the "max recommended" size for Flux. This helps restore the details lost due to Kontext VAE encoding/decoding, especially after multiple passes.
</blockquote>

### Image editing via prompting / reference images
- [HiDream-E1](https://github.com/HiDream-ai/HiDream-E1)
- [Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit)
- [DreamO](https://github.com/bytedance/DreamO)
  - [ComfyUI](https://github.com/jax-explorer/ComfyUI-DreamO)
- OmniGen
  - [1st version](https://github.com/VectorSpaceLab/OmniGen)
  - [2nd version](https://github.com/VectorSpaceLab/OmniGen2)
  - [Fork](https://github.com/SanDiegoDude/OmniGen2)
  - [ComfyUI](https://github.com/set-soft/ComfyUI_OmniGen_Nodes)
- [BagelUI](https://github.com/dasjoms/BagelUI)
- [Infinity](https://github.com/FoundationVision/Infinity)
- [Flux ControlNet](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0)
- Flux Kontext
  - [Official docs](https://docs.bfl.ai/guides/prompting_guide_kontext_i2i)
  - [ComfyWF](https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI)

### Inpainting and other
- [Krita plugin for segmentation](https://github.com/Acly/krita-ai-tools)
- [Parallax / VR creation from 2D image](https://github.com/combatwombat/tiefling)
- [Stable Flow image editing (high VRAM needed or CPU offload!)](https://github.com/snap-research/stable-flow)
- [IDM-VTON (Virtual Try-On)](https://github.com/yisol/IDM-VTON)
- [PhotoDoodle (inpainting effects / decoration)](https://github.com/showlab/PhotoDoodle)
- [img2svg](https://huggingface.co/starvector/starvector-8b-im2svg)
- [ICEdit (in-context editing)](https://github.com/River-Zhang/ICEdit)
- [LanPaint (Thinking Inpainting)](https://github.com/scraed/LanPaint)
- [LUT Maker (GPU accelerated color lookup table generator)](https://github.com/o-l-l-i/lut-maker)
- [Insert Anything](https://github.com/song-wensong/insert-anything)


## Video Generation Models

### AnimateDiff
- [Prompt Travel Video2Video Tutorial](https://stable-diffusion-art.com/animatediff-prompt-travel-video2video/)
- [AnimateDiff CLI Prompt Travel](https://github.com/s9roll7/animatediff-cli-prompt-travel)
- [ComfyUI Guide](https://civitai.com/articles/2379)
- [ComfyUI Node Setup](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
- [AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning)
- [AnimateDiff Video Tutorial](https://www.youtube.com/watch?v=Gz9pT2CwdoI)

### CogVideoX
- [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)
- [CogVideoX-5b-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
- [CogView3 (distilled)](https://github.com/THUDM/CogView3)
- [SageAttention (speedup CogVideo gen)](https://github.com/thu-ml/SageAttention)
- I2V workflow
  - [Github](https://github.com/henrique-galimberti/i2v-workflow/blob/main/CogVideoX-I2V-workflow_v2.json)
  - [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1fqy71b/cogvideoxi2v_updated_workflow)
  - [CivitAI](https://civitai.com/models/785908/animate-from-still-using-cogvideox-5b-i2v)
- LORAs
  - [Wallace & Gromit](https://huggingface.co/Cseti/CogVideoX-LoRA-Wallace_and_Gromit)
  - [Arcane](https://huggingface.co/Cseti/CogVideoX1.0-LoRA-Arcane-v1)

### Hunyuan
- [HuggingFace](https://huggingface.co/tencent/HunyuanVideo)
- [ComfyUI workflow](https://civitai.com/models/1092466?modelVersionId=1243590)
- [Multi Lora Node](https://github.com/facok/ComfyUI-HunyuanVideoMultiLora)
- LORAs
  - [Boreal-HL](https://civitai.com/models/1222102/boreal-hl?modelVersionId=1376844)
  - [Misc](https://civitai.com/user/Remade)
- Fine-Tunes
  - [SkyReels v1](https://github.com/SkyworkAI/SkyReels-V1)
  - [SkyReels i2v workflow](https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy)
- [Image2Video](https://huggingface.co/tencent/HunyuanVideo-I2V)
- [Hunyuan-Custom (subject consistency)](https://github.com/Tencent-Hunyuan/HunyuanCustom)

### Wan
- [Collection](https://huggingface.co/Wan-AI)
- LORAs
  - [Steamboat Willie](https://huggingface.co/benjamin-paine/steamboat-willie-14b)
  - [Flat Color](https://huggingface.co/motimalu/wan-flat-color-v2)
  - [Ghibli](https://civitai.com/models/1474964)

### LXTV
- [Github](https://github.com/Lightricks/LTX-Video)
- LORAs
  - [Melting](https://civitai.com/models/1571778?modelVersionId=1778638)


## Other Video Generation / Manipulation Models and Tools

### Avatar talking animation
- [LatentSync](https://github.com/bytedance/LatentSync)
- [LivePortrait](https://github.com/kijai/ComfyUI-LivePortraitKJ)
- [LivePortrait main repo](https://github.com/KwaiVGI/LivePortrait)
- [Live Portrait and Face Expression Tutorial](https://www.youtube.com/watch?v=q6poA8I7tRM)
- [Sonic](https://github.com/smthemex/ComfyUI_Sonic)
- [HunyuanVideo-Avatar](https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar)
- FantasyTalking
  - Github: https://github.com/Fantasy-AMAP/fantasy-talking
  - Tutorial: https://learn.thinkdiffusion.com/fantasytalking-where-every-images-tells-a-moving-story/

### Restoration / upscaling
- [Video face restoration](https://github.com/wangzhiyaoo/SVFR)
- [Video upscaling](https://huggingface.co/SherryX/STAR)
- [Video upscale workflows](https://civitai.com/search/models?modelType=Workflows&sortBy=models_v9&query=upscale%20vid)
- [KEEP (video upscaling)](https://github.com/jnjaby/KEEP)
- [Video2x](https://github.com/k4yt3x/video2x)
- [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

### Optimization
- [Sage Attention](https://github.com/woct0rdho/SageAttention)
- [Triton Windows](https://github.com/woct0rdho/triton-windows)
- [Flash Attention Windows Wheels](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)
- [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)
  - [ComfyUI](https://github.com/philipy1219/ComfyUI-TaylorSeer)

### Other
- [Alibaba VGen (video generation projects)](https://github.com/ali-vilab/VGen)
- [Open-Sora](https://github.com/hpcaitech/Open-Sora)
- [ToonCrafter](https://github.com/ToonCrafter/ToonCrafter)
- [LVCD: Reference-based Lineart Video Colorization with Diffusion Models](https://github.com/luckyhzt/LVCD)
- [Genmo - Mochi 1](https://github.com/genmoai/models)
- [Rhymes - AI Allegro](https://huggingface.co/rhymes-ai/Allegro)
- [Nvidia's Cosmos models (txt2video, video2world, ...)](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6)
- [DepthFlow](https://github.com/BrokenSource/DepthFlow)
- [Video crop tool (helpful for training dataset creation)](https://github.com/Tr1dae/HunyClip)
- [Video face swap](https://codeberg.org/Cognibuild/ROOP-FLOYD)
- [Vid2DensePose](https://github.com/Flode-Labs/vid2densepose)
- [MagicAnimate](https://github.com/magic-research/magic-animate)
- MoviiGen (Wan Fine-Tune)
  - https://huggingface.co/ZuluVision/MoviiGen1.1
- [VACE (Reference-to-video, video-to-video)](https://github.com/ali-vilab/VACE)
- [Wan2GP (app for using different video models)](https://github.com/deepbeepmeep/Wan2GP)
- [Video Colorization ComfyUI Nodes](https://github.com/jonstreeter/ComfyUI-Deep-Exemplar-based-Video-Colorization)
- [ComfyUI-Video-Depth-Anything](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything)
- [Spline-Path-Control](https://github.com/WhatDreamsCost/Spline-Path-Control/)

## Image Segmentation / Object Detection
- [Blog around image detection techniques](https://blog.roboflow.com/)
- [YOLOv10 (Real-time object detection)](https://github.com/THU-MIG/yolov10 (https://github.com/ultralytics/ultralytics))
- Meta's Segment Anything (SAM)
  - [SAM2](https://github.com/facebookresearch/segment-anything-2)
  - [SAM2.1 Release](https://github.com/facebookresearch/sam2)
  - [RobustSAM (Fine-tuning for improved segmentation of low-quality images)](https://github.com/robustsam/RobustSAM)
  - [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
  - [SAM HQ](https://huggingface.co/syscv-community/sam_hq_vit_b)
- [ViTPose (pose detection)](https://huggingface.co/collections/usyd-community/vitpose-677fcfd0a0b2b5c8f79c4335)
- Real-time object detection
  - [Roboflow leaderboard](https://leaderboard.roboflow.com/)
  - [DEIM](https://github.com/ShihuaHuang95/DEIM)
  - [D-FINE](https://huggingface.co/collections/ustc-community/d-fine-68109b427cbe6ee36b4e7352)
  - DAB-DETR
    - https://huggingface.co/IDEA-Research/dab-detr-resnet-50
    - https://huggingface.co/docs/transformers/main/en/model_doc/dab-detr
  - RT-DETRv3
    - https://github.com/clxia12/RT-DETRv3
    - https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2
    - https://huggingface.co/PekingU/rtdetr_r18vd
  - [RF-DETR (Roboflow)](https://github.com/roboflow/rf-detr)
  - [Frigate (NVR With Realtime Object Detection for IP Cameras)](https://github.com/blakeblackshear/frigate)
  - [Vision-LSTM (technique and models)](https://github.com/NX-AI/vision-lstm)
- Depth estimation
  - https://huggingface.co/apple/DepthPro-hf
  - https://huggingface.co/docs/transformers/main/en/model_doc/depth_pro
  - https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes


## 3D Generation / Manipulation Models and Tools
- [WildGaussians](https://github.com/jkulhanek/wild-gaussians/)
- [Apple's Depth Pro](https://github.com/apple/ml-depth-pro)
- [InstantMesh (img -> 3D Mesh)](https://github.com/TencentARC/InstantMesh)
- Microsoft's Trellis
  - [Official github](https://github.com/microsoft/TRELLIS)
  - [One-Click Installer](https://github.com/IgorAherne/trellis-stable-projectorz)
- [Stable Point Aware 3D](https://huggingface.co/stabilityai/stable-point-aware-3d)
- Hunyuan3D
  - https://github.com/Tencent-Hunyuan/Hunyuan3D-2
  - https://huggingface.co/tencent/Hunyuan3D-2mini
- [Pippo (turnaround view of person)](https://github.com/facebookresearch/pippo)
- [TripoSG (image-to-3D)](https://github.com/VAST-AI-Research/TripoSG)
- [Roblox cube (text-to-3D)](https://github.com/Roblox/cube)
- [LegoGPT (text-to-Lego)](https://github.com/AvaLovelace1/LegoGPT)
- [Step1X-3D (image-to-3D)](https://github.com/stepfun-ai/Step1X-3D)
