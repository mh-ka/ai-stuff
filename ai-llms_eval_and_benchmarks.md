# LLMs - Evaluation / Leaderboards / Comparisons

## Links
- [How to construct domain-specific LLM evaluation systems](https://hamel.dev/blog/posts/evals/)
- [Big Code Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)
- [Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [Coding Benchmark](https://prollm.toqan.ai/leaderboard)
- [CanAiCode Leaderboard](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results)
- [OpenCompass 2.0](https://github.com/open-compass/opencompass)
- [Open LLM Progress Tracker](https://huggingface.co/spaces/andrewrreed/closed-vs-open-arena-elo)
- [MMLU Pro](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro)
- [LLM API Performance Leaderboard](https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard)
- [SEAL Leaderboard](https://scale.com/leaderboard)
- [Lightweight Library from OpenAI to evaluate LLMs](https://github.com/openai/simple-evals)
- [Open LLM Leaderboard 2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- [Skeleton Key Jailbreak](https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique)
- [LLM Eval simplified](https://www.philschmid.de/llm-evaluation)
- [LiveBench](https://livebench.ai/)
- [Benchmark Aggregator](https://benchmark-aggregator-lvss.vercel.app/)
- [LLM Comparison](https://artificialanalysis.ai/)
- [LLM Planning Benchmark](https://github.com/karthikv792/LLMs-Planning)
- [LLM Cross Capabilities](https://github.com/facebookresearch/llm-cross-capabilities)
- [MLE-Bench (Machine Learning tasks benchmark)](https://github.com/openai/mle-bench)
- [Compliance EU AI Act Framework](https://github.com/compl-ai/compl-ai)
- [Open Financial LLM Leaderboard](https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard)
- [Aider Leaderboards](https://aider.chat/docs/leaderboards/)
- [BigCodeBench](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard)
- [BIG-bench](https://github.com/google/BIG-bench)
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [FACTS Leaderboard](https://www.kaggle.com/facts-leaderboard)
- [Massive Text Embedding Benchmark Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Reasoning challenges prompts](https://github.com/cpldcpu/MisguidedAttention)
- [Evaluation model](https://huggingface.co/Unbabel/XCOMET-XL)
- [UGI (Uncensored) Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)
- [GPU Poor LLM Arena](https://huggingface.co/spaces/k-mktr/gpu-poor-llm-arena)
- [Several LLM benchmarks](https://github.com/lechmazur)
- [Creative writing benchmark](https://eqbench.com/creative_writing.html)
- [Agent leaderboard](https://huggingface.co/spaces/galileo-ai/agent-leaderboard)
- [LLM Pricing](https://huggingface.co/spaces/philschmid/llm-pricing)
- [Vision Models Survey](https://nanonets.com/blog/bridging-images-and-text-a-survey-of-vlms/)
- [Context Size Evals](https://github.com/NVIDIA/RULER)
- [Model context sizes](https://github.com/taylorwilsdon/llm-context-limits)
- [LLM translation comparison](https://nuenki.app/blog/llm_translation_comparison)
- [Pydantic Evals](https://ai.pydantic.dev/evals)
- [Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard)
- [Healthbench](https://github.com/m42-health/healthbench/)
- [SWE Rebench](https://swe-rebench.com/leaderboard)
- [Roocode Evals](https://roocode.com/evals)
- [SuperGPQA](https://github.com/SuperGPQA/SuperGPQA)
- [Confabulations](https://github.com/lechmazur/confabulations)
- [PGG Multi-Agent Bench](https://github.com/lechmazur/pgg_bench/)
- [Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [NYT Connections Extended](https://github.com/lechmazur/nyt-connections/)
- [VLMs in social/physical context](https://opensocial.world/leaderboard)
- [Time series forecasting eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
- [LM Eval framework by Google](https://github.com/google/lmeval)
- [VideoGameBench for VLMs](https://github.com/alexzhang13/videogamebench)
- [Finance agent](https://github.com/vals-ai/finance-agent)


## Benchmark List
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
