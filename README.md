# CLIP Model Benchmark — Image Classification

A systematic benchmark comparing all nine OpenAI CLIP model variants on a
custom-labeled image dataset across nine real-world categories. Built as part
of the research phase for [Smart File Sorter](https://github.com/your-username/smart-file-sorter),
a content-aware local file organiser.

This repo also includes a companion benchmark of **sentence-embedding models
for text-document classification** (PDF/DOCX/TXT), extending the same
zero-shot, prompt-ensemble approach to non-image files. See
[Text Model Benchmark — Document Classification](#text-model-benchmark--document-classification)
below.

---

## What this benchmarks

Nine CLIP model variants from OpenAI, compared head-to-head on the same
dataset and the same category prompts:

| Model            | Architecture       | Patch Size | Input Resolution |
| ---------------- | ------------------ | ---------- | ---------------- |
| `RN50`           | ResNet             | —          | 224 × 224        |
| `RN101`          | ResNet             | —          | 224 × 224        |
| `RN50x4`         | ResNet (scaled)    | —          | 288 × 288        |
| `RN50x16`        | ResNet (scaled)    | —          | 384 × 384        |
| `RN50x64`        | ResNet (scaled)    | —          | 448 × 448        |
| `ViT-B/32`       | Vision Transformer | 32 × 32 px | 224 × 224        |
| `ViT-B/16`       | Vision Transformer | 16 × 16 px | 224 × 224        |
| `ViT-L/14`       | Vision Transformer | 14 × 14 px | 224 × 224        |
| `ViT-L/14@336px` | Vision Transformer | 14 × 14 px | 336 × 336        |

Each model is evaluated using **ensemble prompt averaging** — four prompt
phrasings per category are encoded and averaged into a single embedding,
which consistently outperforms a single prompt on zero-shot classification.

---

## Dataset

| Category      | Images  |
| ------------- | ------- |
| Animals       | 70      |
| Code Snippets | 70      |
| Documents     | 70      |
| Food          | 70      |
| Infographics  | 70      |
| Memes         | 69      |
| Real People   | 70      |
| Receipts      | 70      |
| Wallpapers    | 70      |
| **Total**     | **629** |

**The dataset is not included in this repository.** Download it here: [link to your dataset — Google Drive / Kaggle / HuggingFace]

Images are organised into folders by category:

```
dataset/
  animal/
  codesnippet/
  documents/
  food/
  infographic/
  meme/
  real_people/
  receipt/
  wallpapers/
```

---

## Results

Full results are saved in [`benchmark_results.csv`](https://github.com/kindnemo/open_clip_benchmark/blob/main/benchmark_results.csv).
Confusion matrices for all nine models are in [`confusion_matrices.png`](/kindnemo/open_clip_benchmark/blob/main/confusion_matrices.png).
> Results below were produced on a Google Colab T4 GPU instance.
> Inference times will differ on CPU.

| Model    | Overall Accuracy % | Avg ms/image | Conf (correct) | Conf (wrong) |
| -------- | ------------------ | ------------ | --------------- | ------------ |
| RN50     | 87.92              | 13.3         | 0.119           | 0.117        |
| RN101    | 84.74              | 20.9         | 0.119           | 0.117        |
| RN50x4   | 80.76              | 19.3         | 0.119           | 0.116        |
| RN50x16  | 79.17              | 30.2         | 0.119           | 0.117        |
| RN50x64  | 80.45              | 42.2         | 0.120           | 0.118        |
| ViT-B/32 | 86.49              | 13.6         | 0.119           | 0.116        |
| ViT-B/16 | 84.10              | 13.3         | 0.118           | 0.117        |
| ViT-L/14 | 87.12              | 21.0         | 0.120           | 0.118        |

**Selected model for production:**

---

## How to run

### Option 1 — Google Colab (recommended)

1. Open the notebook in Colab:

2. Go to `Runtime → Change runtime type → T4 GPU`.

3. Mount your Google Drive and point `DATASET_PATH` in Cell 2 to your
dataset folder.

4. Run all cells in order (`Runtime → Run all`).

### Option 2 — Local

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

Open `clip_benchmark.ipynb` in Jupyter, set `DATASET_PATH` to your local
dataset folder, and run all cells.
> **Note:** Model weights (~350MB–2GB depending on variant) are downloaded
> from Hugging Face Hub on first run and cached locally. A stable internet
> connection is required the first time.
---

## Metrics explained

**Overall accuracy** — percentage of images where the model's top prediction
matched the true label. The headline number, but not the only one that matters.

**Avg ms/image** — average inference time per image. Measured on a T4 GPU.
On a CPU expect roughly 20-25x slower.

**Conf when correct** — average confidence score on images the model got
right. Higher is better: a model that is right and confident is safe to
deploy against a review threshold.

**Conf when wrong** — average confidence score on images the model got wrong.
Lower is better: a model that is wrong but confident will silently misfile
files without triggering the review folder. This is the number to minimise,
not just overall accuracy.

**Confusion matrix** — a grid showing which categories are being confused
with each other. Each row is the true label, each column is the predicted
label. Numbers on the diagonal are correct predictions; numbers off the
diagonal are mistakes. Darker off-diagonal cells indicate pairs of categories
the model consistently confuses.

---

## Prompt design

Each category uses four prompt phrasings averaged into a single embedding
(ensemble prompting). This consistently outperforms a single prompt on
zero-shot tasks without any model training.

The prompts were designed with the specific confusion pairs in this dataset
in mind:

- **Meme vs Real people** — memes frequently contain photographs of real
people. The meme prompts heavily emphasise the text overlay as the
distinguishing feature.
- **Document vs Receipt** — both are text-heavy. The prompts differentiate
on visual structure: receipts are narrow and columnar with prices;
documents contain paragraph prose with no pricing.
- **Animal vs Wallpaper** — the hardest pair in this dataset. A
high-resolution animal photo can legitimately belong in either category.
The wallpaper prompts emphasise widescreen framing and deliberate artistic
composition as the differentiating features.
- **Infographic vs Document** — both text-heavy. Prompts separate on the
presence of icons, diagrams, and visual data encoding vs. continuous prose.

All prompts are defined in Cell 3 of the notebook and can be modified
independently of the benchmark logic.

---

## Repository structure

```
.
├── clip_benchmark.ipynb              the CLIP image benchmark notebook
├── miniLM_benchmark.ipynb            the text/document benchmark notebook
├── requirements.txt                  Python dependencies
├── benchmark_results.csv             full per-image results for all CLIP models
├── confusion_matrices.png            confusion matrix heatmaps for all CLIP models
├── text_model_confusion_matrices.png confusion matrix heatmaps for all text models
├── .gitignore
└── README.md
```

---

## Requirements

```
open-clip-torch
torch==2.7.1
Pillow
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

The text-document benchmark additionally requires:

```
sentence-transformers
pdfplumber
python-docx
```

Install with:

```
pip install -r requirements.txt
```

For GPU inference (strongly recommended for benchmarking all nine models),
use the CPU-only torch wheel:

```
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Context

This benchmark was conducted to select the best CLIP variant for [Smart File Sorter](https://github.com/your-username/smart-file-sorter),
a local desktop application that watches a Downloads folder and
automatically sorts files into category folders using content-based
classification. The selected model will be used for zero-shot image
classification at inference time on the end user's machine with no GPU.

The primary selection criterion was not highest overall accuracy but **lowest confident-wrong rate** — because the app already has a
confidence-threshold review system that catches uncertain predictions,
the dangerous failure mode is a model that misclassifies with high
confidence, not one that misclassifies with low confidence.

---

## Text Model Benchmark — Document Classification

A systematic benchmark comparing four `sentence-transformers` embedding
models on a custom-labeled document dataset across four real-world
categories. Built as part of the research phase for [Smart File Sorter](https://github.com/your-username/smart-file-sorter),
a content-aware local file organiser.

### What this benchmarks

Four general-purpose `sentence-transformers` models, compared head-to-head
on the same document dataset and the same category prompts:

| Model                  | Params | Embedding Dim |
| ----------------------- | ------ | -------------- |
| `all-MiniLM-L6-v2`      | 22M    | 384            |
| `all-MiniLM-L12-v2`     | 33M    | 384            |
| `all-mpnet-base-v2`     | 109M   | 768            |
| `all-distilroberta-v1`  | 82M    | 768            |

Each model uses the same **ensemble prompt averaging** method as the CLIP
benchmark: four prompt phrasings per category are encoded and averaged into
a single embedding, and each document is classified by cosine similarity
against those four category embeddings.

### Dataset

| Category  | Documents |
| --------- | --------- |
| General   | 95        |
| Invoice   | 81        |
| Legal     | 100       |
| Resume    | 99        |
| **Total** | **375**   |

Documents are PDF, DOCX, and TXT files organised into folders by category:

```
document_dataset/
  general/
  invoice/
  legal/
  resume/
```

Of 382 source files, 7 were skipped after text extraction returned no usable
content (scanned/image-only PDFs with no text layer) and are excluded from
the results above.

Only the first 2,000 characters of each document are extracted and passed
to the model, which is enough signal for this classification task while
keeping inference fast.

### Results

Full confusion matrices for all four models are in
[`text_model_confusion_matrices.png`](https://github.com/kindnemo/open_clip_benchmark/blob/main/text_model_confusion_matrices.png).
> Results below were produced on a Google Colab T4 GPU instance.
> Inference times will differ on CPU.

| Model                  | Overall Accuracy % | Avg ms/doc | Conf (correct) | Conf (wrong) | Conf gap |
| ----------------------- | ------------------ | ---------- | --------------- | ------------ | -------- |
| all-MiniLM-L12-v2       | 79.20               | 93.0       | 0.384           | 0.175        | 0.209    |
| all-distilroberta-v1    | 71.20               | 494.9      | 0.370           | 0.256        | 0.114    |
| all-mpnet-base-v2       | 69.07               | 837.1      | 0.354           | 0.246        | 0.108    |
| all-MiniLM-L6-v2        | 65.87               | 88.4       | 0.350           | 0.230        | 0.120    |

**Selected model: `all-MiniLM-L12-v2`** — it leads on every axis that
matters here: highest accuracy, lowest inference time by a wide margin
(runs at roughly 1/9th the latency of `all-mpnet-base-v2`), and by far the
largest confidence gap of the four, meaning it's both the most accurate and
the most reliably self-aware when it's wrong.

Per-category, all models struggle most on **general** (precision is high
but recall is weak — general documents get misclassified into other
categories, most often invoice or resume) while **legal** is the easiest
category across the board (precision and recall both above 0.87 for every
model). This mirrors the CLIP benchmark's document/receipt confusion pair:
"general" is the catch-all category with no consistent structural signal,
so it absorbs misclassifications from more distinctively-formatted
categories.

### Prompt design

As with the CLIP prompts, each category uses four phrasings designed around
this dataset's specific confusion risks:

- **Invoice vs Legal** — both can contain numbered clauses and formal
language. Invoice prompts emphasise itemised pricing, quantities, and
totals; legal prompts emphasise binding obligations and signature blocks
with no pricing structure.
- **Resume vs General** — both are prose-heavy without pricing or clause
numbering. Resume prompts emphasise the structured sections (work history,
education, skills) that distinguish it from an ordinary letter or memo.
- **General** — deliberately defined as a residual category (correspondence,
memos, reports) rather than by a single positive signal, since it has no
consistent document structure of its own.

All prompts are defined in the notebook and can be modified independently
of the benchmark logic.

### How to run

Open [`miniLM_benchmark.ipynb`](https://github.com/kindnemo/open_clip_benchmark/blob/main/miniLM_benchmark.ipynb)
in Google Colab or Jupyter, point `DATASET_PATH` at your document dataset
folder, and run all cells in order. Model weights are downloaded from
Hugging Face Hub on first run.



