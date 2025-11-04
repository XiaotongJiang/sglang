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
    - Uses FP8 dynamic quantization for attention linear layers
    - Supports FP8 KV cache quantization
    """

    def __init__(
        self,
        is_checkpoint_mxfp4_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        apply_fp8_to_attention: bool = True,
        kv_cache_dtype: str = "auto",
    ):
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.apply_fp8_to_attention = apply_fp8_to_attention
        self.kv_cache_dtype = kv_cache_dtype
        
        log_info_on_rank0(
            logger,
            f"Hybrid quantization config: MXFP4 for MoE, FP8 for attention layers. "
            f"FP8 attention enabled: {apply_fp8_to_attention}",
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
            apply_fp8_to_attention=True,
        )

    def is_static_cfg(self):
        return self.is_checkpoint_mxfp4_serialized

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """
        Returns the appropriate quantization method for each layer type:
        - LinearBase (attention layers): FP8LinearMethod
        - FusedMoE: Mxfp4MoEMethod (from checkpoint)
        - RadixAttention: Fp8KVCacheMethod (for FP8 KV cache)
        """
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        # Handle MoE layers - keep in MXFP4
        if isinstance(layer, FusedMoE):
            if self.is_checkpoint_mxfp4_serialized:
                log_info_on_rank0(logger, f"Using MXFP4 for MoE layer: {prefix}")
                return Mxfp4MoEMethod(prefix=prefix)
            else:
                return Mxfp4DynamicQuantMoEMethod()

        # Handle Linear layers (attention projections) - use FP8
        if isinstance(layer, LinearBase):
            # Check if this layer should be skipped
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            
            # Skip FP8 on HIP for now
            if _is_hip:
                return UnquantizedLinearMethod()
            
            # Apply FP8 to attention layers
            if self.apply_fp8_to_attention:
                log_info_on_rank0(logger, f"Using FP8 for attention layer: {prefix}")
                # Create a minimal FP8 config for the linear method
                from sglang.srt.layers.quantization.fp8 import Fp8Config
                fp8_config = Fp8Config(
                    is_checkpoint_fp8_serialized=False,
                    activation_scheme=self.activation_scheme,
                    ignored_layers=[],
                )
                return Fp8LinearMethod(fp8_config)
            else:
                return UnquantizedLinearMethod()

        # Handle KV cache quantization for attention
        if isinstance(layer, RadixAttention):
            from sglang.srt.layers.quantization.fp8 import Fp8KVCacheMethod, Fp8Config
            
            if self.kv_cache_dtype in ["fp8_e4m3", "fp8_e5m2"]:
                log_info_on_rank0(logger, f"Using FP8 KV cache for attention: {prefix}")
                fp8_config = Fp8Config(
                    is_checkpoint_fp8_serialized=False,
                    activation_scheme=self.activation_scheme,
                )
                return Fp8KVCacheMethod(fp8_config)

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


