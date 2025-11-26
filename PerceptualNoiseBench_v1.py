#!/usr/bin/env python3
"""Hobbled variant of PerceptualNoiseBench.

The intent is educational: the scaffolding runs end-to-end, but key functions
are deliberately left blank so practitioners have to implement the perceptual
maths themselves.  The pipeline therefore emits trivially perturbed images until
the TODO sections are filled in.

Disclaimer: for lawful research only.  By running this file you agree to supply
your own implementations responsibly and to accept full liability for any
downstream effects. Contact the maintainer if you require the full,
fully-instrumented version with the completed perceptual stack.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

INPUT_IMAGE_PATH = "input.png"
OUTPUT_IMAGE_PATH = "output_hobbled.png"
EPSILON = 0.5
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_LEVEL = logging.INFO

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerceptualReport:
    """Minimal perceptual report, intentionally sparse."""

    noise_mean: float
    noise_std: float


def configure_logging() -> None:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def deterministic_rng(seed: int = 123) -> np.random.Generator:
    return np.random.default_rng(seed)


class HobbledPerturber:
    """Perturber shell with missing internals."""

    def __init__(self, epsilon: float = EPSILON) -> None:
        self.epsilon = float(epsilon)
        self.rng = deterministic_rng()

    def load(self, path: str) -> Tuple[Image.Image, np.ndarray]:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img).astype(np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expecting HxWx3 image")
        return img, arr

    def perturb(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        LOGGER.info("Running hobbled perturbation (placeholders still empty).")
        mask = self._contrast_mask(arr[:, :, 0])
        if mask is None:
            LOGGER.warning("Contrast mask not implemented; using uniform weights.")
            mask = np.ones(arr.shape[:2], dtype=np.float32)
        mask = mask[:, :, None]
        noise = self.rng.uniform(-self.epsilon, self.epsilon, arr.shape).astype(
            np.float32
        )
        noise *= mask
        candidate = self._embed_watermark(arr, mask)
        if candidate is None:
            LOGGER.warning("Watermark embedder not implemented; skipping.")
            candidate = arr
        try:
            candidate = self._apply_phase_tricks(candidate)
        except NotImplementedError:
            LOGGER.warning("Phase tricks missing; returning untouched candidate.")
        out = np.clip(candidate + noise, 0, 255)
        return out, noise

    def _contrast_mask(self, luminance: np.ndarray) -> np.ndarray | None:
        """Return a contrast mask; left blank on purpose."""
        pass

    def _embed_watermark(self, arr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Hide a watermark in-place; implement me."""
        pass

    def _apply_phase_tricks(self, arr: np.ndarray) -> np.ndarray:
        """Apply Fourier-domain tricks; replace with actual math."""
        raise NotImplementedError("Phase perturbation left as an exercise.")


def run(
    input_path: str = INPUT_IMAGE_PATH, output_path: str = OUTPUT_IMAGE_PATH
) -> PerceptualReport:
    configure_logging()
    perturber = HobbledPerturber()
    _, arr = perturber.load(input_path)
    perturbed, noise = perturber.perturb(arr)
    Image.fromarray(perturbed.astype(np.uint8)).save(output_path)
    LOGGER.info("Hobbled output written to %s", output_path)
    return PerceptualReport(
        noise_mean=float(noise.mean()), noise_std=float(noise.std())
    )


if __name__ == "__main__":
    REPORT = run()
    LOGGER.info("Noise summary mean=%.6f std=%.6f", REPORT.noise_mean, REPORT.noise_std)
