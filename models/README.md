# models/

Trained model checkpoints and the persistent RAG vector store.

These files are NOT committed to the repo (they're in `.gitignore`) because they're large. They're produced by running the training notebooks and the knowledge-base build script.

## Files created by running the notebooks / scripts

- **`plantdoc_resnet50.pt`** — the main fine-tuned ResNet-50 checkpoint. Produced by `notebooks/02_training.ipynb`. Used by `app.py` at inference time. ~100 MB.
- **`test_predictions.npz`** — saved test-set predictions (y_true, y_pred, probs) from the best model. Produced by `notebooks/02_training.ipynb`. Used by `notebooks/04_error_analysis.ipynb` so you don't have to re-run inference. ~5 MB.
- **`ablation_frozen.pt`**, **`ablation_finetuned.pt`**, **`ablation_noaug.pt`** — checkpoints from each ablation variant. Produced by `notebooks/03_ablation_study.ipynb`. ~100 MB each.
- **`chroma_db/`** — persistent ChromaDB vector store for the RAG pipeline (38 embedded disease documents). Produced by `python -m src.build_knowledge_base`. ~15 MB.

## Downloading pretrained checkpoints

If you want to skip training, pretrained checkpoints are available at:

- `plantdoc_resnet50.pt`: [add your Drive/Box link after uploading]

Place downloaded files here and skip straight to step 5 in `SETUP.md`.
