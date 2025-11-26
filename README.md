# PerceptualNoiseBench

PerceptualNoiseBench is a controlled research environment for generating and auditing perceptual perturbations in images. It is built for reproducible experiments, forensic traceability, and empirical evaluation of human-perceptual invariance versus model-internal feature drift.

The project exists to support work of the following kind:

* Stress-testing watermarking schemes.
* Evaluating feature-fingerprint robustness.
* Quantifying how image classifiers respond to structured perturbations that are imperceptible to humans.
* Reproducing “looks identical to a human, diverges to a model” phenomena with tightly controlled noise fields.
* Creating audit-grade artefacts for research papers, replication packages, and forensic lab notebooks.

Everything in the pipeline is explicit, deterministic, and inspectable.

---

## Purpose

Modern neural networks are highly sensitive to structured spectral changes, chroma rotations, and localized phase shifts that are invisible or insignificant to human observers. This gap between human perception and model sensitivity is both a security risk and a research opportunity.

PerceptualNoiseBench provides a fully documented, deterministic laboratory for studying this gap. It enables researchers to:

* Trace the exact provenance of each manipulation.
* Measure perceptual fidelity (PSNR, SSIM) alongside noise statistics.
* Inspect noise fields, magnified and unclipped, as standalone artefacts.
* Reproduce results across hardware, seeds, and experimental conditions.

This makes it suitable for inclusion in academic replication packages, internal robustness evaluations, and watermark verification workflows.

---

## Capabilities

The pipeline applies a sequence of transformations designed to preserve human visual appearance while modifying spectral content and local structure:

* **Lab-space contrast-masked noise**
  Noise is weighted by luminance gradients so perturbations hide under natural texture.

* **Spread-spectrum watermark embedding**
  A deterministic watermark, derived from a SHA256 hash of the text payload, is mixed into luminance through spectral shaping.

* **Chroma rotation**
  Chroma channels are rotated along iso-luminance arcs to shift features off classifier axes without visible color shifts.

* **Global Fourier phase jitter**
  High-frequency phase components are perturbed, disrupting CNN-favorite local cues.

* **Block-wise frequency scrambling**
  Local patches undergo controlled spectral perturbation to reduce feature alignment without altering macro appearance.

* **Noise field export**
  The absolute noise is magnified and saved to reveal microscopic structure invisible in the perturbed output.

All transformations operate on float32 arrays to remove dtype-related artefacts.

---

## Determinism and Reproducibility

Reproducible perturbation research demands exact traceability. The system enforces:

* Deterministic RNG streams isolated per experiment.
* SHA256 hashing of files, arrays, perturbations, and noise fields.
* Explicit ordering of all major operations.
* Absence of hidden global state.
* Transparent configuration surfaces.

Residual nondeterminism is limited to external library FFT implementations if present; all array hashes are logged so this can be detected.

---

## Outputs

Running the benchmark produces:

* `output.png` – the perturbed image.
* `diff.png` – magnified noise visualization.
* Logged SHA256 hashes for:

  * Input file
  * Input array
  * Perturbed output
  * Noise field
* Logged perceptual metrics:

  * PSNR
  * SSIM
  * Noise mean
  * Noise standard deviation
  * SNR (dB)

These artefacts are suitable for research documentation, forensic evidence trails, or inclusion in supplementary material.

---

## Pipeline Summary

`run()` performs:

1. Logging configuration.
2. Deterministic RNG initialization.
3. Input image loading into float32 RGB arrays.
4. Perturbation via pixel-uniform noise or Lab-space contrast-masked noise.
5. Metric computation.
6. Saving of perturbed image and diff map.
7. Returning `(PSNR, SSIM)` for programmatic evaluation.

The function is intentionally explicit and linear, allowing it to be pasted into a research appendix without commentary.

---

## Installation

```
pip install numpy pillow scikit-image
```

---

## Usage

```
python perceptual_noise_bench.py
```

Place an `input.png` in the working directory or point the script at a different file.

---

## Example Research Applications

* Evaluating whether spread-spectrum watermarks persist under specific perturbation regimes.
* Measuring cross-model sensitivity using PSNR/SSIM-matched images.
* Testing classifier robustness against Fourier-domain or chroma-domain structure shifting.
* Generating replicable adversarial-like examples without employing gradient methods.
* Creating forensic audit trails showing exactly how an image has been transformed.

The system is intentionally transparent and reproducible so researchers can build reliable comparisons and defend their methodology in peer review.

---

## License and Use

PerceptualNoiseBench is designed for legitimate research on perceptual perturbations, watermarking, and forensic analysis. Users are responsible for ensuring compliance with all legal and ethical guidelines when applying these methods to real assets.

---
