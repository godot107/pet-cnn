# TODO

## Phase 1: local (done 2026-10-14 unless unchecked)
- [x] Data: Oxford annotations + HF parquet images; cache; pHash audit (43 pairs, 24 cross-split)
- [x] Classic baseline (HOG + colour histogram, logistic regression) on official + unseen-breeds
- [x] SmallCNN on CPU: official 91.2% bal, unseen-breeds 87.2%, leaky-2023 98.8% (bug worth ~7 pts)
- [x] Executed walkthrough notebook (general audience)
- [x] ResNet-18 (ImageNet-pretrained) locally on CPU (33 s/epoch): official 97.5% bal, unseen-breeds 92.7%
- [x] Per-breed unseen: Sphynx ~75% for both CNNs (hairless); others 82-100%
- [ ] Reboot (GPU driver mismatch: kernel 580.173.02 vs library 580.178), then `scripts/train_scratch_gpu.sh`
      (from-scratch SmallCNN, GPU-only, 60 epochs) and compare with the CPU run; 224 px later?
- [ ] Breed task (37 classes) on the official split
- [ ] Error analysis: the 20 most confident mistakes (notebook section)

## Phase 2: AWS setup (training stays local; `sm-train` is optional and unused)
- [ ] `aws login`; deploy_cost.sh, deploy_lab.sh (bucket holds model.tar.gz for serving)
- [ ] Is ImageNet pretraining "cheating" on breeds? (ImageNet-1k has many of them as classes)

## Phase 3: serving
- [ ] Verify the inference DLC tag; deploy_serve.sh; `sm-invoke` and curl with the API key
- [ ] Measure cold vs warm latency; tune memory_mb
- [ ] Custom domain pets.willieman.com (ACM validation CNAME + target CNAME at Spaceship)
- [ ] Small static page (upload a photo → result) on the custom domain? Key would be public: keep the quota
- [ ] Teardown verified (down.sh), CloudBurn `discover` for leftovers
- [ ] Optional: SageMaker Studio domain stack (idle shutdown) to run the notebook in the cloud

## Write-up
- [ ] Blog: the one-line bug, the duplicate audit, unseen breeds, Azure → AWS cost story
- [ ] Trello: move the card, link the repo
