# Multimodal NLP & Clustering Pipeline

An end-to-end unsupervised learning pipeline spanning text, vision, and multimodal domains. The project clusters Steam game reviews by sentiment and genre, extracts and clusters visual features from flower images using VGG16 transfer learning, classifies Pokemon types from images using CLIP zero-shot inference, and surfaces genre-sentiment correlations from a normalized SQLite database.

## Parts

| Part | Domain | Task |
|------|---------|------|
| 1 | Text (Steam Reviews) | Discover review length structure without labels |
| 2 | Text (Steam Reviews) | Infer game genres from aggregated positive reviews |
| 3 | Vision (TF-Flowers) | Cluster flower images using VGG16 transfer features |
| 4 | Multimodal (Pokemon) | CLIP-based image-text retrieval and type classification |
| 5 | SQL Analytics | Relational analysis of the Steam dataset |

### Part 1 - Review Length Discovery

Constructs pseudo-labels (short = bottom 25% by token count, long = top 25%) and evaluates whether unsupervised clustering can recover that structure from TF-IDF and MiniLM sentence embeddings alone. Benchmarks K-Means and Agglomerative clustering across SVD, UMAP, and no dimensionality reduction.

### Part 2 - Unsupervised Game Genre Structure

Aggregates positive reviews per game into a single document and clusters games by what players praise. Uses a TensorFlow autoencoder and HDBSCAN to discover complaint and praise themes for a held-out game. A small Qwen1.5 LLM generates descriptive labels for each discovered cluster.

### Part 3 - VGG16 Feature Extraction and Flower Clustering

Extracts 4096-dim features from the TF-Flowers dataset using a truncated VGG16 pre-trained on ImageNet. Benchmarks DR methods (SVD, UMAP, deep autoencoder) and clustering algorithms (K-Means, Agglomerative, HDBSCAN) against ground-truth flower class labels. Includes an MLP classifier trained on VGG16 features as a supervised comparison.

### Part 4 - Pokemon CLIP Multimodal Retrieval and Classification

Uses CLIP (ViT-L/14) to retrieve Pokemon images matching type-based text queries and classify each Pokemon's primary type zero-shot. Re-ranks CLIP's top-5 candidates using Qwen3-VL to improve Acc@1, achieving an 11pp gain (33% to 44%).

### Part 5 - SQL Analytics on Steam Reviews

Ingests the Steam reviews dataset into a normalized SQLite schema with three tables: `reviews`, `games`, and a `game_genres` bridge table that splits comma-separated genre tags into individual rows. All aggregations are done in pure SQL using window functions (NTILE, RANK with PARTITION BY), CTEs, and JOINs.

## Setup

**Prerequisites:** Python 3.10+, a CUDA-capable GPU (recommended: 16GB+ VRAM for VGG16 extraction and Qwen3-VL).

Install all dependencies by running the first cell in `multimodal_clustering.ipynb`, or manually:

```bash
pip install sentence-transformers umap-learn hdbscan tensorflow seaborn
pip install torch torchvision transformers datasets
pip install git+https://github.com/openai/CLIP.git kagglehub plotly scipy
```

## Data

The notebook fetches all datasets automatically:

- **Steam reviews** (`main.csv`, `heldout.csv`): loaded from Google Drive when running in Colab, or from a local `data/` directory otherwise. Set `USE_COLAB = True/False` and update the path variables at the top of the configuration cell accordingly.
- **TF-Flowers**: downloaded from TensorFlow's example images URL and cached to `flowers_features_and_labels.npz` after first extraction.
- **Pokemon images**: downloaded via `kagglehub` from the `hlrhegemony/pokemon-image-dataset` dataset.
- **Pokemon CSV**: fetched from a public GitHub repository URL defined in the configuration cell.

## Usage

Open `multimodal_clustering.ipynb` and run cells in order from top to bottom.

**Step 1 - Install and import:** Run the setup cells to install packages and load all imports.

**Step 2 - Configure paths:** In the configuration cell (Section 0), set `USE_COLAB` and update the Steam review file paths to match your environment.

**Step 3 - Run parts sequentially:** Each part is self-contained under its own section header. Parts 1 and 2 depend on the Steam reviews DataFrame loaded in Part 1. Parts 3 and 4 are independent. Part 5 depends on the same DataFrame as Parts 1-2.

## Configuration

Key parameters are defined in the configuration cell (Section 0):

| Parameter | Default | Description |
|---|---|---|
| `USE_COLAB` | `True` | Toggle between Colab/Drive and local file paths |
| `RANDOM_STATE` | `42` | Global random seed for reproducibility |
| `VGG_FEATURES_PATH` | `flowers_features_and_labels.npz` | Cache path for extracted VGG16 features |
| `DB_PATH` | `steam_reviews.db` | Output path for the SQLite database |
| `POKEMON_CSV_URL` | GitHub URL | Source for the Pokemon metadata CSV |

## Output

- `part1_clustering_viz.png` - Side-by-side scatter plots of ground truth vs. cluster assignments for review length
- `part3_tsne_vgg.png` - t-SNE visualization of VGG16 features colored by flower class
- `clip_retrieval_{type}.png` - Top-5 retrieval results for each Pokemon type query
- `steam_reviews.db` - Normalized SQLite database with reviews, games, and game_genres tables
- `part5_sql_viz.png` - Bar chart of genre recommendation rates and review length vs. helpfulness plot

## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) - Contrastive Language-Image Pretraining (Radford et al., 2021)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) - Vision-language model for re-ranking
- [Qwen1.5](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) - LLM used for cluster labelling
- [POPE](https://huggingface.co/datasets/hlrhegemony/pokemon-image-dataset) - Pokemon image dataset via Kaggle
- [TF-Flowers](http://download.tensorflow.org/example_images/flower_photos.tgz) - Flower image dataset via TensorFlow
- [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Sentence embeddings via sentence-transformers
