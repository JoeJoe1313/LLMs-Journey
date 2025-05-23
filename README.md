# LLMs-Journey

The **LLMs-Journey** repository covers topics including:

- **Large Language Models (LLMs)**
- **Agentic Systems and Workflows**
- **Fine-Tuning (e.g., LoRA, QLoRa)**
- **Retrieval-Augmented Generation (RAG)**
- **Vision-Language Models (VLMs)**
- **Complementary Resources and Research**

The projects are organized into topic-specific directories.

## Repository Structure

```
LLMs-Journey/
├── .gitignore
├── README.md
├── requirements.txt
├── simple_fine_tune_lora_mlx.ipynb
├── Agents/
│   ├── fine_tune_func_calling.ipynb
│   ├── hf_unit2.ipynb
│   ├── llamaindex_agentic_workflows.ipynb
│   ├── llamaindex_agents.ipynb
│   ├── llamaindex_hf_course.ipynb
│   ├── llamaindex_tools.ipynb
│   └── llamaindex_with_mlx.ipynb
├── LLMs/
│   ├── gemma_3.ipynb
│   ├── hf_to_mlx_lm_model.ipynb
│   ├── mlx_lm_simple_generate.ipynb
│   ├── qwen2_5_math.ipynb
│   └── reason.py
├── RAG/
│   └── *(TBD)*
├── VLMs/
│   └── *(TBD)*
└── images/
    └── *(Supporting images)*
```

## Detailed File and Directory Descriptions

### Root Directory Files (temp)

- **simple_fine_tune_lora_mlx.ipynb**  
   - A Jupyter Notebook that demonstrates a straightforward fine-tuning example using the LoRA (Low-Rank Adaptation) method within an MLX setup.
   - Blog post: [Fine-Tuning LLMs with LoRA and MLX-LM](https://medium.com/@levchevajoana/fine-tuning-llms-with-lora-and-mlx-lm-c0b143642deb)


### Agents Folder

- Blog post: [Fine-Tuning LLMs for Function-Calling with MLX-LM](https://medium.com/@levchevajoana/fine-tuning-a-model-for-function-calling-with-mlx-lm-d00d587e2559)

### LLMs Folder

TBD

### RAG Folder

TBD

### VLMs Folder

- Blog post: [Qwen2.5-VL with MLX-VLM](https://medium.com/@levchevajoana/qwen2-5-vl-with-mlx-vlm-c4329b40ab87)
- Blog post: [Image Segmentation with PaliGemma 2 Mix and MLX](https://medium.com/@levchevajoana/image-segmentation-with-paligemma-2-mix-and-mlx-7e69e077968b)
- Blog post: [Image Segmentation with PaliGemma 2 mix, Transformers, Docker, FastAPI, and GitHub Actions](https://medium.com/@levchevajoana/image-segmentation-with-paligemma-2-mix-transformers-docker-fastapi-and-github-actions-ff6d00253832)

### images Folder

- **images/**  
  Contains visual resources used throughout the repository.


## What to read?

Books:
- [Foundations of Large Language Models](https://arxiv.org/pdf/2501.09223)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/index)
- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

Agents:
- [Hugging Face Agents Course](https://huggingface.co/agents-course)
- [ai-agents-for-beginners (Microsoft)](https://github.com/microsoft/ai-agents-for-beginners)
- [Agents (Google's whitepaper)](https://www.kaggle.com/whitepaper-agents)
- [Agency Is Frame-Dependent](https://arxiv.org/abs/2502.04403)

Promptning:
- [Prompt Engineering (Google's whitepaper)](https://www.kaggle.com/whitepaper-prompt-engineering)
- [GPT-4.1 Prompting Guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)

Fine-Tuning:
- [🤗 PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)

RAG related:
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Chain-of-Retrieval Augmented Generation](https://arxiv.org/pdf/2501.14342)
- [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

Evaluation:
- [A Survey on LLM-as-a-Judge](https://arxiv.org/pdf/2411.15594)
- [ARC Prize 2024: Technical Report](https://arxiv.org/pdf/2412.04604)
- [FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI](https://arxiv.org/pdf/2411.04872)
- [MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations](https://arxiv.org/abs/2502.06453v1)

Datasets:
- [NuminaMath 1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)

Models:
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)
- [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864) and [Round and Round We Go! What makes Rotary Positional Encodings useful?](https://arxiv.org/pdf/2410.06205)
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file) and [paper](https://arxiv.org/abs/2502.13923)
- [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math?tab=readme-ov-file) and [paper](https://arxiv.org/abs/2409.12122)
- [PaliGemma 2: A Family of Versatile VLMs for Transfer](https://arxiv.org/abs/2412.03555v1)
- [Magma: A Foundation Model for Multimodal AI Agents](https://arxiv.org/abs/2502.13130)
- [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242)
- [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491)
- [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](https://arxiv.org/abs/2504.10479)

Chain-of-Thought
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

Visualisation-of-Thought:
- [Imagine while Reasoning in Space: Multimodal Visualization-of-Thought](https://arxiv.org/pdf/2501.07542)

Test-Time Scaling
- [s1: Simple test-time scaling](https://github.com/simplescaling/s1) and [paper](https://arxiv.org/abs/2501.19393)

Test-Time Compute
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)

AlphaGeometry:
- [Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2](https://arxiv.org/abs/2502.03544)

Apple:
- [Machine Learning Research at Apple](https://machinelearning.apple.com)
- [Distillation Scaling Laws](https://arxiv.org/abs/2502.08606)
- [Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models](https://arxiv.org/abs/2501.12370)
- [Reversal Blessing: Thinking Backward May Outpace Thinking Forward in Multi-choice Questions](https://arxiv.org/abs/2502.18435v2)

Misc:
- [Scaling Pre-training to One Hundred Billion Data for Vision Language Models](https://arxiv.org/abs/2502.07617)
- [Competitive Programming with Large Reasoning Models](https://arxiv.org/abs/2502.06807)
- [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)
- [Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model](https://arxiv.org/abs/2502.10248), [repo](https://github.com/stepfun-ai/Step-Video-T2V)
- [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)
- [Deepseek Papers](https://huggingface.co/collections/Presidentlin/deepseek-papers-674c536aa6acddd9bc98c2ac)
- [InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU](https://arxiv.org/abs/2502.08910)
- [Leveraging the true depth of LLMs](https://arxiv.org/abs/2502.02790)
- [NOLIMA: Long-Context Evaluation Beyond Literal Matching](https://arxiv.org/pdf/2502.05167)
- [Memory Layers at Scale](https://arxiv.org/abs/2412.09764)
- [Towards an AI co-scientist](https://arxiv.org/abs/2502.18864)
- [On the consistent reasoning paradox of intelligence and optimal trust in AI: The power of 'I don't know'](https://arxiv.org/abs/2408.02357)
- [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387)
- [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)
- [How new data permeates LLM knowledge and how to dilute it](https://arxiv.org/abs/2504.09522)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Resources

MLX:
- [mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main)
- [ml-explore](https://github.com/ml-explore)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

LangChain:
- [Build a Retrieval Augmented Generation (RAG) App: Part 1](https://python.langchain.com/docs/tutorials/rag/)

Vector Databases:
- [Chroma](https://www.trychroma.com/home)
- [Pinecone](https://www.pinecone.io)
- [Weaviate](https://weaviate.io)

HuggingFace:
- [MLX Community](https://huggingface.co/mlx-community)
- [Using MLX at Hugging Face](https://huggingface.co/docs/hub/en/mlx)

Leonie Notebooks:
- [Fine-tuning Gemma 2 JPN for Yomigana with LoRA](https://www.kaggle.com/code/iamleonie/fine-tuning-gemma-2-jpn-for-yomigana-with-lora)
- [Advanced RAG with Gemma, Weaviate, and LlamaIndex](https://www.kaggle.com/code/iamleonie/advanced-rag-with-gemma-weaviate-and-llamaindex)
- [RAG with Gemma on HF 🤗 and Weaviate in DSPy](https://www.kaggle.com/code/iamleonie/rag-with-gemma-on-hf-and-weaviate-in-dspy)

GitHub Repos:
- [rag-cookbooks](https://github.com/athina-ai/rag-cookbooks)
- [Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)
- [Multimodal-RAG-Implementation](https://github.com/CornelliusYW/Multimodal-RAG-Implementation)
- [Data_Science_Learning_Material](https://github.com/CornelliusYW/Data_Science_Learning_Material)
- [RAG-To-Know](https://github.com/CornelliusYW/RAG-To-Know)
- [Docling](https://github.com/DS4SD/docling?tab=readme-ov-file)

Blogs/Posts:
- [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)
- [Transformers from Scratch](https://www.brandonrohrer.com/transformers)
- [Chain of Thought](https://www.k-a.in/cot.html)
