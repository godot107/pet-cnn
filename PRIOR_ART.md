# Prior art (scanned 2026-10-14)

Scanned with `kaggle-projects/tools/prior_art_scan.py` for the Kaggle `dogs-vs-cats` competition and
the `tanlikesmath/the-oxfordiiit-pet-dataset` dataset; the code of the top notebook from each was read.

| Votes | Notebook | Split / evaluation |
|---|---|---|
| 4,706 | [uysimty/keras-cnn-dog-or-cat-classification](https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification) | Random 80/20 `train_test_split`; the 20% drives `ReduceLROnPlateau` / early stopping **and** is the reported score. The competition test set is unlabeled, so no held-out number exists. |
| 873 | [bulentsiyah/…vgg16-fine-tuning](https://www.kaggle.com/code/bulentsiyah/dogs-vs-cats-classification-vgg16-fine-tuning) | ImageNet VGG16 fine-tune; validation reported as the result. |
| 157 | [tanlikesmath/…with-fastai](https://www.kaggle.com/code/tanlikesmath/oxford-iiit-pet-dataset-classification-with-fastai) | ImageNet ResNet-50, `fit_one_cycle`; the validation set doubles as the test set. 37-breed task. |

Most-voted titles advertise 95–99% ("val_acc=+98.5%", "Class-f1 (0.992)"), all on validation sets
that also chose the stopping point.

## Synthesis

- **Consensus baseline:** a small Keras CNN from scratch (~80–90% on the Kaggle set), or an
  ImageNet-pretrained backbone fine-tuned (97–99%).
- **Common gaps:** no test set separate from validation; random splits with no near-duplicate check;
  accuracy only (no per-class recall on an imbalanced 1:2 cat:dog set); no "unseen category" test.
  ImageNet-1k already contains many of these breeds as classes, so pretrained transfer partly
  re-uses labels it was trained on; few notebooks mention this.
- **Adopted:** augmentation (shift + flip), early stopping, a transfer-learning arm (ResNet-18).
- **Differentiated angle:**
  1. three-way train/val/test with the test set used once;
  2. a perceptual-hash duplicate audit (24 official train/test pairs found and removed from training);
  3. an **unseen-breeds** hard benchmark;
  4. the 2023 bug kept as an explicit leaky contrast;
  5. production path on AWS: SageMaker spot training, a serverless endpoint, an API-key-gated API.
- **Credit / license:** Oxford-IIIT Pet, Parkhi, Vedaldi, Zisserman & Jawahar, *Cats and Dogs*,
  CVPR 2012, CC BY-SA 4.0. Images are downloaded at runtime, never committed. No code copied
  from the notebooks above.
