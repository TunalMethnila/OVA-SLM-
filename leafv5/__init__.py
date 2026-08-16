"""LEAFv5: A Rapidly Adapting, Ultra-Efficient Architecture for Small Language Models.

Implementation of the architecture described in the LEAFv5 paper:
  * Identity-start residual highways (per-channel scales init to 0)
  * Multi-scale depthwise local path (kernels 3, 5, 9, 15)
  * Stabilized Multi-Timescale Delta Memory (Fast / Medium / Slow plasticity heads)
  * Linear complexity, recurrent inference with near-constant memory
"""

from .config import ModelConfig, PRESETS
from .model import (LeafLM, LeafBlock, MultiTimescaleDeltaV2, MultiScaleLocalPath,
                    RMSNorm, SlidingWindowAttention, MoEFFN)

__version__ = "0.6.0"
__all__ = ["ModelConfig", "PRESETS", "LeafLM", "LeafBlock", "MultiTimescaleDeltaV2", "MultiScaleLocalPath", "RMSNorm"]
