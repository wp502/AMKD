# Adaptive Multimodal Knowledge Distillation via Synergistic Multiknowledge Supervision


## 🏗️ Project Structure

```

├── datasets/                 # Dataset loaders
├── distillers/               # Distillation methods (MSD, DSMD, HRAD, etc.)
├── models/                   # Student models
├── raw-models/               # Saved checkpoints
├── evaluation.py             # Evaluation (classification & retrieval)
├── load_data.py              # Data loading utilities
├── load_model.py             # Model loading
├── main.py                   # Main training entry
├── train.py                  # Training loop
├── utils.py                  # Utilities
├── zoo_pretrain_teacher.py   # Teacher pretraining
├── zoo_eval_teacher.py       # Teacher evaluation
└── run.txt                   # Training scripts
```

---

## ⚙️ Requirements

* Python >= 3.8
* PyTorch >= 1.10
* CUDA (recommended)

Install dependencies:

```bash
pip install torch torchvision tqdm open_clip_torch
```

---

## 📊 Supported Tasks

* **Multi-label Classification**

  * MMIMDb

* **Image-Text Retrieval**

  * Flickr30K
  * MS-COCO

---

## 🧠 Teacher Model Pretraining

### Multi-label Classification (MMIMDb)

```bash
python zoo_pretrain_teacher.py \
  --dataset mmimdb \
  --teacher_model clip-RN101 \
  --batch_size 128 \
  --learning_rate 5e-5
```

---

### Retrieval (Flickr30K)

```bash
python zoo_pretrain_teacher.py \
  --dataset flickr-30k \
  --teacher_model clip-RN101 \
  --batch_size 256 \
  --learning_rate 5e-5
```

---

### Retrieval (MS-COCO)

```bash
python zoo_pretrain_teacher.py \
  --dataset ms-coco \
  --teacher_model clip-ViT-B-16 \
  --batch_size 256 \
  --learning_rate 5e-5
```

---

## 📈 Teacher Evaluation

```bash
python eval_teacher.py \
  --dataset flickr-30k \
  --teacher_model clip-RN101 \
  --ckpt raw_models/teachers/train/clip-RN101_flickr-30k_best.pth
```

---

## 🎓 Student Baselines

### Example (MMIMDb)

```bash
python main.py \
  --dataset mmimdb \
  --student_model clip-RN50 \
  --distiller none \
  --epoch 30 \
  --learning_rate 5e-5
```

---

### Example (MS-COCO)

```bash
python main.py \
  --dataset ms-coco \
  --student_model clip-ViT-B-32 \
  --distiller none \
  --epoch 30 \
  --learning_rate 1e-5 \
  --batch_size 256
```

---

## 🔥 Distillation (HRAD)

Our proposed method: **HRAD (Hierarchical Relational Adaptive Distillation)**

### Example (MMIMDb)

```bash
python main.py \
  --dataset mmimdb \
  --teacher_model_1 clip-ViT-B-16 \
  --teacher_model_2 clip-ViT-L-14 \
  --student_model clip-ViT-B-32 \
  --distiller hrad \
  --epoch 20 \
  --learning_rate 5e-5
```

---

### Example (MS-COCO)

```bash
python main.py \
  --dataset ms-coco \
  --teacher_model_1 clip-ViT-B-16 \
  --teacher_model_2 clip-ViT-L-14 \
  --student_model clip-RN50 \
  --distiller hrad \
  --epoch 40 \
  --learning_rate 1e-5 \
  --batch_size 128
```

---

### Example (Flickr30K)

```bash
python main.py \
  --dataset flickr-30k \
  --teacher_model_1 clip-ViT-B-16 \
  --teacher_model_2 clip-ViT-L-14 \
  --student_model clip-ViT-B-32 \
  --distiller hrad \
  --epoch 40 \
  --learning_rate 5e-5 \
  --project_dim 64
```

---

## 🧩 Supported Distillation Methods

* `none` — No distillation (baseline)
* `msd`
* `dsmd`
* `kdmcse`
* `g2d`
* `dclip`
* `hrad` (ours)

---

## 📌 Key Features

* Multi-teacher knowledge distillation
* Adaptive loss weighting
* Cross-modal alignment
* Supports multiple architectures:

  * CLIP variants
  * ViT-BERT
  * ResNet-BERT

---

## 📁 Checkpoints

* Saved under:

```
raw_models/
```

---

## 📝 Citation

Coming soon...


