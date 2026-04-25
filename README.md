# 🌿 PlantPulse - AI Plant Disease Detector + Treatment Advisor

A computer-vision and RAG-powered assistant that identifies plant diseases from leaf photos and retrieves grounded treatment advice from a curated knowledge base of 38 plant-disease conditions.

## What it Does

PlantPulse is a Streamlit web app that helps home gardeners figure out what's wrong with their plants. You upload a photo of a sick leaf, and the app gives you a diagnosis, shows you which part of the leaf the model focused on, and pulls up actual treatment advice you can use.

Under the hood, it's a fine-tuned ResNet-50 trained on the PlantVillage dataset (about 87,000 images covering 38 plant-disease combinations) for the diagnosis. Then I added Grad-CAM on top so users can see a heatmap of where the model was looking when it made the prediction, which matters for trust. The treatment advice comes from a RAG pipeline I built with LangChain and ChromaDB, pulling from a knowledge base I wrote covering all 38 disease classes. GPT-4o-mini handles the final answer synthesis, but the actual facts come from the knowledge base, not from whatever the LLM picked up during training.

There's also a chat interface for follow-up questions, scoped to whatever disease was just diagnosed. The reason I went with RAG instead of just calling an LLM directly is that vague AI answers aren't useful for someone who actually needs to treat a plant. The KB has specific fungicide names, resistant cultivar names, and temperature ranges that favor outbreaks. RAG keeps the answers grounded in those specifics.

## Quick Start

This project runs locally via Streamlit. Follow these steps:

1. **Clone the repo and install dependencies**
```bash
   git clone https://github.com/joeylesun/plantPulse.git
   cd plantPulse
   pip install -r requirements.txt
```

2. **Download the PlantVillage dataset** (needed for training only; skip if you're just running inference with a pre-trained checkpoint). See `data/README.md` for links and extraction steps. Place it under `data/plantvillage/color/`.

3. **Configure your OpenAI API key.** Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`.

4. **Train the model** (or download the pretrained checkpoint linked in `models/README.md`). Open `notebooks/02_training.ipynb` and run all cells. This saves `models/plantdoc_resnet50.pt`. On a single GPU this takes roughly 1-2 hours for 10 epochs. On CPU expect closer to 8 hours, which is what I used.

5. **Build the RAG vector store.**
```bash
   python -m src.build_knowledge_base
```
   This embeds the 38 disease documents in `data/disease_knowledge_base.json` into a persistent ChromaDB at `models/chroma_db/`.

6. **Launch the app.**
```bash
   streamlit run app.py
```
   Open http://localhost:8501 in your browser, upload a leaf photo, and you're off.

For detailed installation troubleshooting, GPU setup, and deployment notes, see **[SETUP.md](SETUP.md)**.

## Video Links

- **Demo Video** (3-5 min, non-technical pitch): [add link after upload]
- **Technical Walkthrough** (5-10 min, for ML engineers): [add link after upload]

## Evaluation

### Vision model

Fine-tuned ResNet-50 on PlantVillage (about 87K images across 38 classes), 70/15/15 train/val/test split, 10 epochs, Adam optimizer with cosine LR schedule.

| Model variant | Test accuracy |
|---|---|
| Random baseline (1/38) | 2.68% |
| ResNet-50, frozen backbone (2 epochs) | 95.69% |
| ResNet-50, fine-tuned, no augmentation (2 epochs) | 97.70% |
| ResNet-50, fine-tuned + augmented (2 epochs) | 97.15% |
| **ResNet-50, fine-tuned + augmented (4 epochs, final)** | **98.06%** |

The final fine-tuned model hits 98.06% test accuracy, a 95.4 percentage point improvement over the random baseline. Fine-tuning gives a clear ~1.5pp boost over the frozen-backbone variant, showing that adapting the ResNet-50 backbone to plant disease features earns its compute cost.

The augmentation result was a surprise. On PlantVillage's clean, controlled test set (uniform backgrounds, studio lighting, single leaves per image), the no-augmentation variant marginally beat the augmented one (97.70% vs 97.15%). This goes against the textbook claim that augmentation always helps, but it makes sense for this dataset specifically: train, val, and test are drawn from nearly the same distribution, so aggressive augmentation introduces noise without much regularization benefit. I kept augmentation in the final model anyway because the Grad-CAM analysis (notebook 05) shows the model relies partly on background features. For real-world photos with cluttered backgrounds and varied lighting, augmentation should help generalization. See `notebooks/03_ablation_study.ipynb` for the full discussion.

Full training curves, confusion matrix, and per-class accuracy plots are saved under `docs/` and documented in `notebooks/02_training.ipynb` and `notebooks/04_error_analysis.ipynb`.

### Error analysis

The most-confused class pairs (see `notebooks/04_error_analysis.ipynb`) are biologically plausible. The top confusion is between Corn Northern Leaf Blight and Cercospora Gray Leaf Spot (both brown elongated lesions on corn leaves), followed by Tomato Early Blight vs Late Blight (same plant, similar rot patterns), and Tomato Early Blight vs Target Spot (both have concentric-ring lesions). Real plant pathologists confuse the same pairs, which suggests the model has learned actual disease morphology rather than memorizing backgrounds.

Grad-CAM visualizations in `notebooks/05_gradcam_visualization.ipynb` partially confirm this. For 4 out of 6 sample diseases (apple scab, grape black rot, potato late blight, tomato early blight) the heatmap concentrates on the lesions. For two cases (tomato late blight and corn common rust) the heatmap drifts onto the background, which is a real concern for deployment on wild photos and motivates the augmentation choice above.

### RAG pipeline

Evaluated on a 10-question golden set (`notebooks/06_rag_evaluation.ipynb`) comparing PlantPulse-RAG against vanilla GPT-4o-mini without retrieval. Grounding score = fraction of expected key phrases (specific fungicide names, variety names, temperature ranges) present in the answer.

| System | Mean grounding score |
|---|---|
| Vanilla GPT-4o-mini | 73.3% |
| **PlantPulse RAG** | **83.3%** |

RAG outperforms vanilla by 10 percentage points. The gap is smaller than I expected because GPT-4o-mini has clearly absorbed a lot of horticultural knowledge during pretraining (extension service docs, gardening sites, Wikipedia), so for common diseases like apple scab it can name specific fungicides without retrieval. Where RAG clearly wins is on questions where vanilla hedges or refuses to be specific, like "Is there a cure for citrus greening?" - vanilla responds with "as of my last knowledge update..." while RAG gives a direct, grounded answer because the KB contains the fact explicitly.

There was one failure case worth flagging: when asked about resistant apple scab varieties, RAG returned "I don't have that information" even though the KB document explicitly lists Liberty, Enterprise, and Freedom. The class filter likely didn't apply correctly for that query, or the strict-context prompt caused the LLM to refuse rather than synthesize. This is a real limitation worth fixing before deployment.

The bigger story beyond the raw 10pp delta is consistency. RAG answers stay grounded in documented facts, while vanilla sometimes drifts into vague advice. For a real deployment helping growers make treatment decisions, that consistency probably matters more than the absolute score.

## Individual Contributions

This project was completed individually by **Joey Sun**. All components, including data processing, CNN training and ablation, Grad-CAM implementation, RAG pipeline design, knowledge-base authoring, and Streamlit application, were designed and built by the author. AI-assisted code authoring is documented in [ATTRIBUTION.md](ATTRIBUTION.md).