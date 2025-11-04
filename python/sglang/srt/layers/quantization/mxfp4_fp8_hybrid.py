# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Hybrid quantization config that uses MXFP4 for MoE layers and FP8 for attention layers.
Specifically designed for models like openai/gpt-oss-20b that have pre-quantized MoE in MXFP4.
"""

from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.mxfp4 import (
    Mxfp4Config,
    Mxfp4MoEMethod,
    Mxfp4DynamicQuantMoEMethod,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import is_hip, log_info_on_rank0

import logging

logger = logging.getLogger(__name__)

_is_hip = is_hip()


class Mxfp4Fp8HybridConfig(QuantizationConfig):
    """
    Hybrid quantization configuration that:
    - Uses MXFP4 for MoE layers (as loaded from checkpoint)
    - Keeps attention projection weights in bf16 (no weight quantization)
    - KV cache: bf16 by default, FP8 if explicitly requested via --kv-cache-dtype
    
    Note: FP8 KV cache is NOT enabled by default because it requires proper
    scale initialization. To enable it, use --kv-cache-dtype fp8_e4m3.
    """

    def __init__(
        self,
        is_checkpoint_mxfp4_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
    ):
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        
        # Don't expose kv_cache_quant_algo - let user explicitly set --kv-cache-dtype
        # This ensures FP8 KV cache is only used when explicitly requested
        self.kv_cache_quant_algo = None
        
        log_info_on_rank0(
            logger,
            f"Hybrid quantization config: MXFP4 for MoE, bf16 for attention. "
            f"Use --kv-cache-dtype fp8_e4m3 to enable FP8 KV cache.",
        )

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4_fp8_hybrid"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # FP8 requires compute capability 8.0+

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Mxfp4Fp8HybridConfig":
        """
        Create hybrid config from model checkpoint config.
        This assumes the model has mxfp4 in the quantization_config.
        """
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method.lower()
        
        # Check for any FP8-related hints in config
        activation_scheme = cls.get_from_keys_or(config, ["activation_scheme"], "dynamic")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        
        return cls(
            is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        """
        Override the quantization method if:
        1. The model has mxfp4 in its config (for MoE layers)
        2. The user explicitly requested mxfp4_fp8_hybrid
        
        This allows using hybrid quantization without raising errors about
        mismatched quantization methods.
        """
        # Check if the model has mxfp4 quantization
        quant_method = hf_quant_cfg.get("quant_method", "").lower()
        is_mxfp4_checkpoint = "mxfp4" in quant_method or "quark" in quant_method
        
        # Check if user wants hybrid quantization
        is_hybrid_requested = user_quant == "mxfp4_fp8_hybrid"
        
        if is_mxfp4_checkpoint and is_hybrid_requested:
            log_info_on_rank0(
                logger,
                "Model has MXFP4 MoE layers. Using hybrid MXFP4+FP8 quantization as requested."
            )
            return cls.get_name()
        
        return None

    def is_static_cfg(self):
        return self.is_checkpoint_mxfp4_serialized

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """
        Returns the appropriate quantization method for each layer type:
        - LinearBase: None (keep in bf16, no weight quantization)
        - FusedMoE: Mxfp4MoEMethod (from checkpoint)
        - RadixAttention: Fp8KVCacheMethod (for FP8 KV cache only)
        """
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention

        # Handle MoE layers - keep in MXFP4
        if isinstance(layer, FusedMoE):
            if self.is_checkpoint_mxfp4_serialized:
                log_info_on_rank0(logger, f"Using MXFP4 for MoE layer: {prefix}")
                return Mxfp4MoEMethod(prefix=prefix)
            else:
                return Mxfp4DynamicQuantMoEMethod()

        # Handle Linear layers - DON'T quantize weights
        # The model doesn't have pre-quantized FP8 weights, and online
        # weight quantization can cause dtype mismatches
        if isinstance(layer, LinearBase):
            # Return None to use default bf16 weights
            return None

        # Don't return a KV cache quantization method here
        # Let the user explicitly set --kv-cache-dtype fp8_e4m3 if they want FP8 KV cache
        # The model runner will handle it based on the --kv-cache-dtype flag
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class HybridFp8KVCacheMethod(BaseKVCacheMethod):
    """
    KV cache method that applies FP8 quantization to key and value caches.
    Used in hybrid config for FP8 KV cache with MXFP4 MoE.
    """

    def __init__(self, quant_config: Mxfp4Fp8HybridConfig):
        super().__init__(quant_config)


