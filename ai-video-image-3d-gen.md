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
- [OmniGen](https://github.com/VectorSpaceLab/OmniGen)

## Image Manipulation with AI
- [BIRD (Image restoration)](https://github.com/hamadichihaoui/BIRD)
- Removal of Backgrounds
  - [RMBG](https://huggingface.co/briaai/RMBG-1.4)
  - [InSPyReNet](https://github.com/plemeri/InSPyReNet)
  - [BiRefNet: https://github.com/ZhengPeng7/BiRefNet (ComfyUI](https://github.com/1038lab/ComfyUI-RMBG))
  - [BEN: https://huggingface.co/PramaLLC/BEN (ComfyUI](https://github.com/DoctorDiffusion/ComfyUI-BEN))
  - [BEN2](https://huggingface.co/PramaLLC/BEN2)
  - [Leaderboard](https://huggingface.co/spaces/bgsys/background-removal-arena)
- Face Swapping
  - [Roop Unleashed](https://codeberg.org/rcthans/roop-unleashednew)
  - [Roop Floyd](https://codeberg.org/Cognibuild/ROOP-FLOYD)
  - [Rope-next](https://github.com/lodgecku/Rope-next)
  - [VisoMaster](https://github.com/visomaster/VisoMaster)
  - FaceFusion (3.0.0)
    - [Github](https://github.com/facefusion/facefusion)
    - [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1fpbm3p/facefusion_300_has_finally_launched/)
  - [ACE++](https://www.patreon.com/posts/face-swapping-121224741)
- Human Animation
  - [Vid2DensePose](https://github.com/Flode-Labs/vid2densepose)
  - [MagicAnimate](https://github.com/magic-research/magic-animate)
- [Krita plugin for segmentation](https://github.com/Acly/krita-ai-tools)
- [Parallax / VR creation from 2D image](https://github.com/combatwombat/tiefling)
- [Stable Flow image editing (high VRAM needed or CPU offload!)](https://github.com/snap-research/stable-flow)
- [IDM-VTON (Virtual Try-On)](https://github.com/yisol/IDM-VTON)
- [MEMO (Expressive Talking)](https://github.com/gjnave/memo-for-windows)
- Upscaler Models and Tools
  - [Siax](https://civitai.com/models/147641/nmkd-siax-cx)
  - [Ultrasharp](https://civitai.com/models/116225/4x-ultrasharp)
  - [Superscale](https://civitai.com/models/141491/4x-nmkd-superscale)
  - [4x-IllustrationJaNai-V1-DAT2](https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2)
  - [Upscayl](https://github.com/upscayl/upscayl)
- [PhotoDoodle (inpainting effects / decoration)](https://github.com/showlab/PhotoDoodle)
- [img2svg](https://huggingface.co/starvector/starvector-8b-im2svg)

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

## Other Video Generation / Manipulation Models and Tools
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
- [Lipsync](https://github.com/bytedance/LatentSync)
- [Video upscaling](https://huggingface.co/SherryX/STAR)
- [Video upscale workflows](https://civitai.com/search/models?modelType=Workflows&sortBy=models_v9&query=upscale%20vid)
- [Live Portrait and Face Expression Tutorial](https://www.youtube.com/watch?v=q6poA8I7tRM)
- [Video face restoration](https://github.com/wangzhiyaoo/SVFR)

## AI Object Detection
- [Blog around image detection techniques](https://blog.roboflow.com/)
- [YOLOv10 (Real-time object detection)](https://github.com/THU-MIG/yolov10 (https://github.com/ultralytics/ultralytics))
- Meta's Segment Anything (SAM)
  - [SAM2](https://github.com/facebookresearch/segment-anything-2)
  - [SAM2.1 Release](https://github.com/facebookresearch/sam2)
  - [RobustSAM (Fine-tuning for improved segmentation of low-quality images)](https://github.com/robustsam/RobustSAM)
  - [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [ViTPose (pose detection)](https://huggingface.co/collections/usyd-community/vitpose-677fcfd0a0b2b5c8f79c4335)
- Real-time object detection
  - [Roboflow leaderboard](https://leaderboard.roboflow.com/)
  - [DEIM](https://github.com/ShihuaHuang95/DEIM)
  - [D-FINE](https://github.com/Peterande/D-FINE)
  - DAB-DETR
    - https://huggingface.co/IDEA-Research/dab-detr-resnet-50
    - https://huggingface.co/docs/transformers/main/en/model_doc/dab-detr
  - RT-DETRv3
    - https://github.com/clxia12/RT-DETRv3
    - https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2
    - https://huggingface.co/PekingU/rtdetr_r18vd
  - [RF-DETR (Roboflow)](https://github.com/roboflow/rf-detr)
- Depth estimation
  - https://huggingface.co/apple/DepthPro-hf
  - https://huggingface.co/docs/transformers/main/en/model_doc/depth_pro

## 3D Generation / Manipulation Models and Tools
- [WildGaussians](https://github.com/jkulhanek/wild-gaussians/)
- [Apple's Depth Pro](https://github.com/apple/ml-depth-pro)
- [InstantMesh (img -> 3D Mesh)](https://github.com/TencentARC/InstantMesh)
- Microsoft's Trellis
  - [Official github](https://github.com/microsoft/TRELLIS)
  - [One-Click Installer](https://github.com/IgorAherne/trellis-stable-projectorz)
- [Stable Point Aware 3D](https://huggingface.co/stabilityai/stable-point-aware-3d)
- Hunyuan3D
  - https://github.com/tencent/Hunyuan3D-2
  - https://huggingface.co/tencent/Hunyuan3D-2mini
- [Pippo (turnaround view of person)](https://github.com/facebookresearch/pippo)
