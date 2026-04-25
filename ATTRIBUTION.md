# ATTRIBUTION.md

## AI-generated code

**Tool used**: Claude (Anthropic), Opus 4.7 model.

**How it was used**: Claude was used as a scaffolding assistant: I described the required components and rubric items, and Claude drafted initial implementations of source files and notebooks. I then reviewed, edited, and tested each piece. I am responsible for the correctness of every line of code submitted.

File-level attribution (where AI scaffolding was applied):

| File | Degree of AI assistance |
|---|---|
| `src/dataset.py` | Code drafted by Claude based on my requirements (5 augmentation techniques, 70/15/15 split, DataLoader). I reviewed and tested the augmentation pipeline, and made minor edits to the augmentation order. |
| `src/model.py` | Code drafted by Claude based on my spec (ResNet-50 with custom classifier head, freeze toggle for ablation). I verified the architecture and parameter counts. |
| `src/train.py` | Code drafted by Claude. I ran the actual training experiments, debugged the device placement issue when switching to MPS, and interpreted the results. |
| `src/baseline.py` | Drafted by Claude, reviewed and edited by me. |
| `src/evaluate.py` | Drafted by Claude, reviewed and edited by me. |
| `src/gradcam.py` | Standard Grad-CAM algorithm from Selvaraju et al. 2017. PyTorch implementation drafted by Claude. I tested it and interpreted the resulting heatmaps, including identifying the resolution-limit failure on Corn Common Rust. |
| `src/predict.py` | Drafted by Claude as a refactor of inline app code. |
| `src/rag.py` | Drafted by Claude. I specified the design — including the class-filter mechanism that scopes retrieval to the CNN's predicted disease — and chose ChromaDB + sentence-transformers over alternatives. |
| `src/utils.py` | Drafted by Claude. The rate limiter, logger setup, and safe-call decorator pattern are standard production patterns; the implementations are AI-drafted but edited in small areas by me. |
| `src/error_analysis.py` | Drafted by Claude, reviewed by me. |
| `data/disease_knowledge_base.json` | Content drafted with Claude from general horticultural knowledge. I cross-referenced against the extension service materials listed below to sanity-check fungicide names, resistant variety names, and temperature ranges. |
| `app.py` | Drafted by Claude. I tested the full UI flow end-to-end and verified prediction, Grad-CAM, RAG advice, and chat all work. Also made various minor feature tweaks myself.|
| `notebooks/*.ipynb` | Notebook structure and code cells drafted by Claude. I ran all six notebooks, debugged issues during runs, interpreted results, and wrote the discussion sections in my own voice based on what I observed. |

I mainly used Claude to accelerate the production of this project, because AI is extremely efficient at writing implementations of methods. I understand each and every bit of code that was AI generated, and debugged and reviewed all pieces myself.

## External libraries and tools

This project uses standard Python libraries listed in `requirements.txt`. The main ones are PyTorch and torchvision (for the CNN), Streamlit (for the web app), LangChain and ChromaDB (for the RAG pipeline), sentence-transformers (for local embeddings), and the OpenAI API (gpt-4o-mini for answer synthesis). All are open-source or used per their standard terms.

The pretrained ResNet-50 weights are the ImageNet weights bundled with torchvision (He et al., "Deep Residual Learning for Image Recognition," 2016).

## Dataset

Trained on the PlantVillage dataset, downloaded from Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Citation: Hughes, D., Salathé, M. (2015). "An open access repository of images on plant health to enable the development of mobile disease diagnostics." arXiv:1511.08060.

## Methods referenced

- **Grad-CAM** (`src/gradcam.py`) implements the standard algorithm from Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017.
- **RAG pattern** is the standard retrieval-augmented generation approach (Lewis et al., NeurIPS 2020).

## Knowledge base sources

Disease information in `data/disease_knowledge_base.json` was cross-referenced against public extension service materials (UC IPM, Cornell, Penn State Extension) for accuracy. Entries are original summaries, not copied text.

## My contribution

I designed the overall architecture (CNN + Grad-CAM + RAG + Streamlit), selected the ablation axes, authored and fact-checked the knowledge base, designed the RAG's class-filter mechanism and prompt, ran and interpreted all experiments, and am responsible for everything submitted. Where AI tools accelerated authoring, I reviewed the output before committing it and take full responsibility for any errors.
