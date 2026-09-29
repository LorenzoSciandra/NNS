# TODO — TMLR submission

Tasks derived from the NeurIPS 2026 reviews (R1, R2, R3) and from the internal review of the paper.
Text-only fixes addressing the reviews have already been applied to the manuscript.

## 0. Before submitting

- [ ] **Dual submission.** If the NeurIPS 2026 decision is still pending, withdraw the paper first. TMLR does not allow concurrent submissions (resubmitting a rejected NeurIPS paper is fine).
- [ ] **Fix inconsistent numbers between Table 1 and the appendix accuracy tables** (`tables/model_split.tex` vs `tables/*_results.tex`). Find which run set is correct and regenerate both tables from the same runs.
  - VGG16 / Fashion-MNIST: %Reduction 74.4% (Table 1) vs 77.23% (Table `tab:fmnist`)
  - VGG16 / CIFAR-10: 94.2% / 24.5% (Table 1) vs 84.06% / 52.0% (Table `tab:cifar10`)
  - VGG16 / CIFAR-100: 85.8% (Table 1) vs 82.07% (Table `tab:cifar100`)
  - VGG11 / CIFAR-10: 88.8% (Table 1) vs 94.2% (Table `tab:cifar10`)
  - Then update the text claims: "up to 94%" / "35.2%–94.2%" reduction (abstract, §3.3, §4.4, Conclusion, Appendix E) and the "7.9%–38.9%" training-fraction range (§3.3).
- [ ] Check that the anonymized repository (anonymous.4open.science) contains no identifying information (author names, paths, git history).

## 1. Mandatory experiments

- [ ] **Depth-reduction baseline (R1, main objection).** Compare NNS with at least one of LaCoOT (Quétu et al., ICCV 2025), Layer Folding (Ben Dror et al., BMVC 2022), or CKA-based layer pruning (Pons et al., ICPR 2024) on shared settings (e.g., ResNet18 / CIFAR-10). Report accuracy, parameters, FLOPs, and training cost.
- [ ] **Comparison with the Tunnel Effect split (R1).** For every configuration, compute the post-hoc TE split (first layer whose linear probe reaches ≥95% of the final accuracy) on the already trained models and report it next to the NNS split, together with the accuracy of the model truncated at that layer. Directly addresses R1's "split layer is often shallower than NNS" point.
- [ ] **Uncertainty everywhere (R2, main concern).**
  - [ ] State in every figure caption whether results are averaged over runs (Figures 2, 3, 5, 6, 7 and Appendices B–D).
  - [ ] Add shaded std bands to the curves.
  - [ ] Extend the tunnel-seed vs optimization-seed study (Table 3, currently 2 seeds × 2 configurations) to 5 seeds and more configurations.
- [ ] **Make the split decision visible (R2).**
  - [ ] Mark the selected split layer in Figure 2 (IFC over epochs) and Figure 3 (IFC per layer).
  - [ ] Add a small plot of each layer's extractor/contractor label over epochs until the stopping epoch.
  - [ ] Fix repeated/similar colors (e.g., layers 0, 9, 10 in Figure 2b); use distinct colors or line styles.
- [ ] **Labeling-rule ablation (R1).**
  - [ ] Sign of the variation w.r.t. epoch 1 (current) vs absolute/relative thresholds on the IFC (cf. Figures 18–20).
  - [ ] Alternative reference value (average over the first k epochs, short warm-up).
  - [ ] Patience sweep: 5 / 10 / 15 / 20 / 30 epochs.
- [ ] **At least one larger-scale result (R1).** ImageNet with ResNet18/50 ideally; otherwise ImageNet-100 or Tiny-ImageNet.

## 2. Strongly recommended (cheap)

- [ ] **Effect of the number of classes (R3).** Train one network on CIFAR-100 subsets with 2, 10, 25, 50, 100 classes; report IFC curves and selected split.
- [ ] **Computational cost.** Report wall-clock training time of NNS vs full training vs baselines, and the per-epoch overhead of computing the IFC on the tunnel set.
- [ ] **Numerical Rank question (R1).** Explain why NR for MLP12 / CIFAR-10 is ~1200 in Figure 15 but <800 in Masarczyk et al. (Figure 2): eigenvalue threshold, tunnel set vs full set, layer width (1024), architecture differences. Add a note in Appendix B.5.
- [ ] **Autoencoder + linear classifier baseline (R2, question).** Either run it (e.g., on CIFAR-10) or add a short justification: NNS uses supervised features selected online, not a separately trained representation.
- [ ] **Regularization hypothesis.** NNS_LP outperforms full training on CIFAR-100 in all configurations; if possible, support the explanation given in §4.4 (e.g., train/test gap of full vs truncated models).

## 3. Rebuttal / text items (already handled in the manuscript, re-check after new results)

- [x] Discuss depth-reduction methods (Pons et al., LaCoOT, Layer Folding) — Related Works.
- [x] Discuss NC / layer-wise collapse works (Zarka et al., Wang et al.), class-discrimination pruning (Liu et al.), early exits (BranchyNet), transfer learning — Related Works.
- [x] Headline results table (Full vs NNS_LP, mean ± std) — Table 2, §4.4.
- [x] IFC formal definition in the main text; simplex ETF definition; EB-LTH description.
- [x] Definition of "simplification" as structural depth reduction (R3).
- [x] Experimental grid explained: every architecture on every dataset, ResNets also on CUB, 21 configurations (R3).
- [x] Labeling rule insensitive to IFC scale / number of classes / layer width (R3).
- [x] NNS is conservative, not the shallowest viable split (R1).
- [x] Limitations: scale, alternative labeling heuristics (R1).
- [ ] After new experiments: update §3.3, §4, Conclusion, and keep the main body ≤ 12 pages.
- [ ] Missing full stop reported by R2 ("line 295" of the NeurIPS version): locate and fix.
