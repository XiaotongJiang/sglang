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

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from enum import IntEnum, auto

import torch
from torch import nn
from transformers import LlamaConfig, PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    ReplicatedLinear
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, make_layers, BumpAllocator, get_bool_env_var, get_int_env_var, is_cuda, is_hip
from sglang.utils import get_exception_traceback
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

_is_hip = is_hip()
_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE = auto()


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x, forward_batch=None):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support llamafy/Qwen-Qwen2.5-7B-Instruct-llamafied with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        # self.self_attn = LlamaAttention(
        #     config=config,
        #     hidden_size=self.hidden_size,
        #     num_heads=config.num_attention_heads,
        #     num_kv_heads=config.num_key_value_heads,
        #     layer_id=layer_id,
        #     rope_theta=rope_theta,
        #     rope_scaling=rope_scaling,
        #     rope_is_neox_style=rope_is_neox_style,
        #     max_position_embeddings=max_position_embeddings,
        #     quant_config=quant_config,
        #     prefix=add_prefix("self_attn", prefix),
        #     bias=attention_bias,
        # )

        # Set default values for LlamaAttentionMLA
        head_dim = config.hidden_size // config.num_attention_heads
        config.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", head_dim)
        config.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        config.v_head_dim = getattr(config, "v_head_dim", head_dim)
        config.kv_lora_rank = getattr(config, "kv_lora_rank", head_dim)

        self.self_attn = LlamaAttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            qk_nope_head_dim_per_head=config.qk_nope_head_dim_per_head,
            qk_rope_head_dim_per_head=config.qk_rope_head_dim_per_head,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=config.kv_lora_rank,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=torch.cuda.Stream(),
        )

        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: Optional[BumpAllocator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Create a zero allocator for LlamaAttentionMLA
        zero_allocator = BumpAllocator(
            buffer_size=2,  # Only need 2 buffers for attention
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LlamaDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            # FIXME(@ying): reduce the number of proxy tensors by not fusing layer norms
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            deferred_norm = None

        # Create a BumpAllocator for LlamaAttentionMLA
        zero_allocator = BumpAllocator(
            buffer_size=len(self.layers) * 2,  # 2 buffers per layer
            dtype=torch.float32,
            device=hidden_states.device,
        )

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn

            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )


class LlamaForCausalLMMLA(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        # ".q_proj": (".qkv_proj", 0),
        # ".k_proj": (".qkv_proj", 1),
        # ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # (".qkv_proj", ".q_proj", "q"),
            # (".qkv_proj", ".k_proj", "k"),
            # (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        self.capture_aux_hidden_states = False

    def _init_model(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return LlamaModel(config, quant_config=quant_config, prefix=prefix)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_hidden_dim(self, module_name):
        # return input_dim, output_dim
        if module_name in ["q_proj", "o_proj", "qkv_proj"]:
            return self.config.hidden_size, self.config.hidden_size
        elif module_name in ["kv_proj"]:
            return self.config.hidden_size, self.config.hidden_size // (
                self.config.num_attention_heads // self.config.num_key_value_heads
            )
        elif module_name == "gate_up_proj":
            return self.config.hidden_size, self.config.intermediate_size
        elif module_name == "down_proj":
            return self.config.intermediate_size, self.config.hidden_size
        else:
            raise NotImplementedError()

    def get_module_name(self, name):
        params_mapping = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
        }
        return params_mapping.get(name, name)

    def get_module_name_from_weight_name(self, name):
        for param_name, weight_name, shard_id, num_shard in self.stacked_params_mapping:
            if weight_name in name:
                return (
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1

    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    def post_load_weights(self, is_nextn=False):

        # Perform post-processing after loading weights
        layer_ids = (
            range(self.config.num_hidden_layers)
            if not is_nextn
            else [self.config.num_hidden_layers]
        )
        for layer_id in layer_ids:
            self_attn = (
                self.model.layers[layer_id].self_attn
                if not is_nextn
                else self.model.decoder.self_attn
            )
            if hasattr(self_attn.kv_b_proj, "qweight"):
                # AWQ compatible
                if _is_cuda:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                    ).T
                else:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                        0,
                        0,
                        0,
                    ).T
            else:
                w = self_attn.kv_b_proj.weight
            # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
            # This may affect the accuracy of fp8 model.
            # Fix deepseek v3 blockwise bmm by using deep_gemm
            use_deep_gemm_bmm = False
            model_dtype = torch.get_default_dtype()

            if w.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                if hasattr(self.quant_config, "weight_block_size"):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        if _is_hip:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=w,
                                weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                                input_scale=None,
                            )
                        else:
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv

                        if (
                            _is_cuda
                            and weight_block_size[0] == 128
                            and weight_block_size[1] == 128
                            and model_dtype == torch.bfloat16
                        ):
                            if _ENABLE_JIT_DEEPGEMM and get_bool_env_var(
                                "SGL_USE_DEEPGEMM_BMM", "false"
                            ):
                                block_scale = weight_scale
                                use_deep_gemm_bmm = True
                            else:
                                w = block_quant_dequant(
                                    weight,
                                    weight_scale,
                                    weight_block_size,
                                    model_dtype,
                                )
                        else:
                            w, scale = block_quant_to_tensor_quant(
                                weight, weight_scale, weight_block_size
                            )
                            self_attn.w_scale = scale
                else:
                    weight = w
                    weight_scale = self_attn.kv_b_proj.weight_scale
                    w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                    self_attn.w_scale = scale

            if w.dtype == torch.int8:
                if hasattr(self.quant_config, "weight_block_size"):
                    # block-wise int8 need it
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv
                        w = int8_block_dequant(
                            weight, weight_scale, weight_block_size
                        ).to(torch.bfloat16)
                else:
                    # channel-wise int8 need it
                    w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                        torch.bfloat16
                    )

            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim * self_attn.num_kv_heads)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim * self_attn.num_kv_heads], dim=1)
            if not use_deep_gemm_bmm:
                w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                w_kc_per_head = w_kc.view(self_attn.num_kv_heads, -1, w_kc.shape[1])
                n_gqa_group = self_attn.num_local_heads // self_attn.num_kv_heads
                w_kc_per_head = torch.repeat_interleave(w_kc_per_head, repeats=n_gqa_group, dim=0)
                self_attn.w_kc = w_kc_per_head.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                    if _is_hip:
                        self_attn.w_scale *= 2.0
                # self._fuse_vc_o_proj_single_gpu(self_attn)
            else:
                num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
                num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
                ws_kc, ws_vc = block_scale.unflatten(
                    0, (-1, (num_tiles_k + num_tiles_n))
                ).split([num_tiles_k, num_tiles_n], dim=1)
                self_attn.w_scale_k = ws_kc.transpose(1, 2).contiguous()
                self_attn.w_scale_v = ws_vc.contiguous()
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous()
                self_attn.w_vc = w_vc.contiguous()
                self_attn.use_deep_gemm_bmm = True

    def _fuse_vc_o_proj_single_gpu(self, self_attn):
        """
        Given a layer's self_attn that already owns
            • self_attn.w_vc           – (H_kv, kv_lora_rank, v_dim)
            • self_attn.o_proj.weight  – (hidden, H_local * v_dim)
        build
            • self_attn.w_fused        – (hidden, H_local * kv_lora_rank)
        so that  matmul(attn_flat, w_fused.T)  replaces the two-step path.
        """
        # ---------- constants ----------
        H_local = self_attn.num_local_heads
        H_kv    = self_attn.num_kv_heads
        g       = H_local // H_kv            # heads per KV group
        Dv      = self_attn.v_head_dim
        L       = self_attn.kv_lora_rank
        H_out   = self_attn.hidden_size

        # ---------- reshape the two source weights ----------
        # o-proj : (H_out, H_local*Dv)  ->  (H_out, H_local, Dv)
        W_o = self_attn.o_proj.weight.view(H_out, H_local, Dv)

        w_vc_per_head = self_attn.w_vc.view(self_attn.w_vc.shape[1], H_kv, -1)
        w_vc_per_head = torch.repeat_interleave(w_vc_per_head, repeats= (H_local // H_kv), dim=-2)
        
        # ---------- contract D_v (the value dim) ----------
        # einsum : (o h d , h d l) -> (o h l)
        W_fused = torch.einsum('ohd,lhd->ohl', W_o, w_vc_per_head) \
                        .reshape(H_out, H_local * L)
        W_fused_T = W_fused.transpose(0, 1)
        self_attn.W_fused_T = W_fused_T

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # (".qkv_proj", ".q_proj", "q"),
            # (".qkv_proj", ".k_proj", "k"),
            # (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Handle FP8 kv-scale remapping
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading kv_scale from ckpts towards new design.
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")
        self.post_load_weights()

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        try:
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                logger.info(
                    "word embedding is tied for this model, return embed_tokens.weight as lm_head.weight."
                )
                return (
                    self.model.embed_tokens.weight.cpu()
                    .to(torch.float32)
                    .numpy()
                    .tolist()[:truncate_size]
                )

            mapped_name = name
            mapped_shard_id = None
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name in name:
                    mapped_name = name.replace(weight_name, param_name)
                    mapped_shard_id = shard_id
                    break
            params_dict = dict(self.named_parameters())
            param = params_dict[mapped_name]
            if mapped_shard_id is not None:
                if mapped_shard_id in ["q", "k", "v"]:
                    num_heads = self.config.num_attention_heads // tp_size
                    num_kv_heads = self.config.num_key_value_heads // tp_size
                    head_dim = (
                        self.config.hidden_size // self.config.num_attention_heads
                    )
                    if mapped_shard_id == "q":
                        offset = 0
                        size = num_heads * head_dim
                    elif mapped_shard_id == "k":
                        offset = num_heads * head_dim
                        size = num_kv_heads * head_dim
                    elif mapped_shard_id == "v":
                        offset = (num_heads + num_kv_heads) * head_dim
                        size = num_kv_heads * head_dim
                    weight = param.data.narrow(0, offset, size)
                elif mapped_shard_id in [0, 1]:
                    intermediate_size = self.config.intermediate_size
                    slice_size = intermediate_size // tp_size
                    if mapped_shard_id == 0:  # gate_proj
                        offset = 0
                        size = slice_size
                    elif mapped_shard_id == 1:  # up_proj
                        offset = slice_size
                        size = slice_size

                    weight = param.data.narrow(0, offset, size)
                else:
                    weight = param.data
            else:
                weight = param.data
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered_weights, weight)
                weight = torch.cat(gathered_weights, dim=1)
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]

        except Exception:
            logger.error(
                f"Error getting weights by name {name} in LlamaForCausalLMMLA: {get_exception_traceback()}"
            )
            return None

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        num_layers = self.config.num_hidden_layers
        self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]


class LlamaAttentionMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        qk_nope_head_dim_per_head: int,
        qk_rope_head_dim_per_head: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim_per_head = qk_nope_head_dim_per_head
        self.qk_rope_head_dim_per_head = qk_rope_head_dim_per_head
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.qk_head_dim_per_head = qk_nope_head_dim_per_head + qk_rope_head_dim_per_head
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.num_kv_heads = num_kv_heads
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim_per_head,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_kv_heads * (self.qk_nope_head_dim_per_head + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_local_heads * self.v_head_dim, # 32 * 64 = 2048
            self.hidden_size, # 2048
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        self.use_mla = True
        self.rotary_emb = get_rope(
            self.v_head_dim // 2 if self.use_mla else self.v_head_dim,
            rotary_dim=self.v_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
            use_mla=self.use_mla,
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb.forward = self.rotary_emb.forward_native

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim_per_head,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
            is_llamamla = True
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim_per_head + self.qk_rope_head_dim_per_head,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
            is_llamamla = True
        )

        self.alt_stream = alt_stream

        self.w_kc = None
        self.w_vc = None
        self.w_scale = None
        self.w_fused_T = None

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False

        self.flashinfer_mla_disable_ragged = global_server_args_dict[
            "flashinfer_mla_disable_ragged"
        ]
        self.disable_chunked_prefix_cache = global_server_args_dict[
            "disable_chunked_prefix_cache"
        ]
        self.attention_backend = global_server_args_dict["attention_backend"]
        self.rocm_fused_decode_mla = get_bool_env_var(
            "SGLANG_ROCM_FUSED_DECODE_MLA", "false"
        )

        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = get_int_env_var(
            "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD", 8192
        )

    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        def _dispatch_mla_subtype():
            if _is_hip:
                if (
                    self.rocm_fused_decode_mla
                    and forward_batch.forward_mode.is_decode()
                ):
                    return AttnForwardMethod.MLA_FUSED_ROPE
                else:
                    return AttnForwardMethod.MLA
            else:
                return AttnForwardMethod.MLA

        if self.attention_backend == "flashinfer":
            # Flashinfer MLA: Do not absorb when enabling ragged prefill
            if (
                not self.flashinfer_mla_disable_ragged
                and forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return _dispatch_mla_subtype()
        elif self.attention_backend == "fa3":
            # Flash Attention: Use MHA with chunked KV cache when prefilling on long sequences.
            if forward_batch.extend_prefix_lens_cpu is not None:
                sum_extend_prefix_lens = sum(forward_batch.extend_prefix_lens_cpu)
            if (
                forward_batch.forward_mode.is_extend()
                and not self.disable_chunked_prefix_cache
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and (
                    sum_extend_prefix_lens >= self.chunked_prefix_cache_threshold
                    or sum_extend_prefix_lens == 0
                )
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            elif forward_batch.forward_mode.is_extend():
                return AttnForwardMethod.MHA
            else:
                return _dispatch_mla_subtype()
        else:
            # Triton: Use normal computation for prefill and use weight absorption for extend/decode
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return _dispatch_mla_subtype()

    def customized_rope(self, positions, q_pe, k_pe, forward_batch):
        # This is a customized rope function for gqa to mla
        # We modify the q_pe and k_pe to include the non rope part
        # then do the rope operation
        # then split the q_pe and k_pe into the rope part and the non rope part
        # only need the rope part
        k_pe = k_pe.view(k_pe.shape[0], self.num_kv_heads, -1)
        if self.use_mla:
            if forward_batch.forward_mode.is_extend():
                return self.rotary_emb.forward(positions, q_pe, k_pe) # TODO: figure out why extend kernal and non kernel are different
            else:
                return self.rotary_emb.forward_cuda(positions, q_pe, k_pe)
        else:
            keep_dim = 16 # TODO: hardcode for now
            def _alloc_scratch(name, like):
                buf = getattr(self, name, None)
                if buf is None or buf.size(0) < like.size(0):
                    buf = torch.empty(
                        like.size(0), like.size(1), like.size(2) * 2,
                        dtype=like.dtype, device=like.device
                    )
                    setattr(self, name, buf)
                return buf[:like.size(0)]

            q_pad = _alloc_scratch('_rope_q_buf', q_pe)
            k_pad = _alloc_scratch('_rope_k_buf', k_pe)

            q_pad[..., :keep_dim]                        = q_pe[..., :keep_dim]
            q_pad[..., 2*keep_dim : 3*keep_dim] = q_pe[..., keep_dim:]

            k_pad[..., :keep_dim]                        = k_pe[..., :keep_dim]
            k_pad[..., keep_dim + keep_dim : keep_dim + 2*keep_dim] = k_pe[..., keep_dim:]
            q_rot, k_rot = self.rotary_emb.forward_cuda(positions, q_pad, k_pad)

            q_pe[..., :keep_dim] = q_rot[..., :keep_dim]
            q_pe[..., keep_dim:] = q_rot[..., 2*keep_dim : 3*keep_dim]

            k_pe[..., :keep_dim] = k_rot[..., :keep_dim]
            k_pe[..., keep_dim:] = k_rot[..., 2*keep_dim : 3*keep_dim]

            return q_pe, k_pe

            k_pe = k_pe.view(k_pe.shape[0], self.num_kv_heads, -1)
            q_customized = torch.cat([q_pe[..., :keep_dim], torch.zeros_like(q_pe[..., keep_dim:]), q_pe[..., keep_dim:], torch.zeros_like(q_pe[..., keep_dim:])], dim=-1)
            k_customized = torch.cat([k_pe[..., :keep_dim], torch.zeros_like(k_pe[..., keep_dim:]), k_pe[..., keep_dim:], torch.zeros_like(k_pe[..., keep_dim:])], dim=-1)
            q_customized, k_customized = self.rotary_emb(positions, q_customized, k_customized)

            q_pe = torch.cat([q_customized[..., :keep_dim], q_customized[..., 2 * keep_dim: 3 * keep_dim]], dim=-1)
            k_pe = torch.cat([k_customized[..., :keep_dim], k_customized[..., 2 * keep_dim: 3 * keep_dim]], dim=-1)
            return q_pe, k_pe

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        return self.forward_core(s)

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states, None, forward_batch, None

        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)

        if attn_forward_method == AttnForwardMethod.MHA:
            inner_state = self.forward_normal_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            inner_state = self.forward_normal_chunked_kv_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA:
            inner_state = self.forward_absorb_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE:
            inner_state = self.forward_absorb_fused_mla_rope_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        else:
            raise NotImplementedError
        return None, attn_forward_method, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, attn_forward_method, forward_batch, inner_state = (
            intermediate_state
        )
        if inner_state is None:
            return hidden_states

        if attn_forward_method == AttnForwardMethod.MHA:
            return self.forward_normal_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            return self.forward_normal_chunked_kv_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA:
            return self.forward_absorb_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE:
            return self.forward_absorb_fused_mla_rope_core(*inner_state)
        else:
            raise NotImplementedError

    def forward_normal_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            all_q = self.q_proj(hidden_states)[0]
            all_q_nope, all_q_pe = all_q.split([self.num_local_heads * self.qk_nope_head_dim_per_head, self.num_local_heads * self.qk_rope_head_dim_per_head], dim=-1)
            all_q_nope = all_q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim_per_head)
            all_q_pe = all_q_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim_per_head)
            q = torch.cat([all_q_nope, all_q_pe], dim=-1)
            # TODO: the way shuffling work is we have 2k, split it into [1k, 1k]
            # [1k ,1k] : [nope, rope]
            # we first split by 1k into 2 tensor of 1, 32, 32
            # then concat on last dim to get 1, 32, 64 => this one we should be good
            # watch out to do the same on the K side (first rope then nope)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        _, q_pe = q.split([self.qk_nope_head_dim_per_head, self.qk_rope_head_dim_per_head], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        # kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        k_nope = kv[..., : self.kv_lora_rank].view(-1, self.num_kv_heads, self.qk_nope_head_dim_per_head)
        v = kv[..., self.kv_lora_rank :].view(-1, self.num_kv_heads, self.v_head_dim)
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        q_pe, k_pe = self.customized_rope(positions, q_pe, k_pe, forward_batch)
        q[..., self.qk_nope_head_dim_per_head :] = q_pe
        k = torch.empty((q.shape[0], self.num_kv_heads, q.shape[2]), device=q.device, dtype=q.dtype)
        k[..., : self.qk_nope_head_dim_per_head] = k_nope
        k[..., self.qk_nope_head_dim_per_head :] = k_pe.reshape(k.shape[0], -1, self.qk_rope_head_dim_per_head)
        # latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe.view(k_pe.shape[0], 1, -1)

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )
        return q, k, v, forward_batch

    def forward_normal_core(self, q, k, v, forward_batch):	    
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False, kv_b_proj = self.kv_b_proj)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)   
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # # overlap qk norm
            # if self.alt_stream is not None and get_is_capture_mode():
            #     current_stream = torch.cuda.current_stream()
            #     self.alt_stream.wait_stream(current_stream)
            #     q = self.q_a_layernorm(q)
            #     with torch.cuda.stream(self.alt_stream):
            #         k_nope = self.kv_a_layernorm(k_nope)
            #     current_stream.wait_stream(self.alt_stream)
            # else:
            #     q = self.q_a_layernorm(q)
            #     k_nope = self.kv_a_layernorm(k_nope)

            k_nope = k_nope.unsqueeze(1)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            # q = self.q_proj(hidden_states)[0].view(
            #     -1, self.num_local_heads, self.qk_head_dim_per_head
            # )
            all_q = self.q_proj(hidden_states)[0]
            q_nope, q_pe = all_q.split(
                [self.num_local_heads * self.qk_nope_head_dim_per_head,
                self.num_local_heads * self.qk_rope_head_dim_per_head],
                dim=-1
            )
            q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim_per_head)
            q_pe   = q_pe  .view(-1, self.num_local_heads, self.qk_rope_head_dim_per_head)           
            # all_q = self.q_proj(hidden_states)[0]
            # all_q_nope, all_q_pe = all_q.split([self.num_local_heads * self.qk_nope_head_dim_per_head, self.num_local_heads * self.qk_rope_head_dim_per_head], dim=-1)
            # all_q_nope = all_q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim_per_head)
            # all_q_pe = all_q_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim_per_head)
            # q = torch.cat([all_q_nope, all_q_pe], dim=-1)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope, k_pe = latent_cache.split(
                [self.kv_lora_rank,
                self.kv_lora_rank],
                dim=-1
            )
            k_nope = k_nope.unsqueeze(1)
            # k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
        # q_nope, q_pe = q.split([self.qk_nope_head_dim_per_head, self.qk_rope_head_dim_per_head], dim=-1)
        if k_pe is None:
            k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)
        else:
            k_pe = k_pe.unsqueeze(1)
        
        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            # w_kc_per_head = self.w_kc.view(self.num_kv_heads, -1, self.w_kc.shape[1])
            # n_gqa_group = self.num_local_heads // self.num_kv_heads
            # q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
            q_nope = torch.einsum('bhd,hdr->bhr', q_nope, self.w_kc)

        # q_nope_out = q_nope_out.transpose(0, 1)

        k_pe = k_pe.view(-1, self.num_kv_heads, self.qk_rope_head_dim_per_head)
        q_pe, k_pe = self.customized_rope(positions, q_pe, k_pe, forward_batch)

        return q_pe, k_pe, q_nope, k_nope, forward_batch, zero_allocator

    def forward_absorb_core(
        self, q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator
    ): 
        if self.attention_backend == "fa3" or self.attention_backend == "flashinfer":
            # k_rope=torch.repeat_interleave(k_pe, repeats= (self.num_local_heads // self.num_kv_heads), dim=1)
            k_rope = k_pe.view(-1, 1, self.qk_rope_head_dim)
            attn_output = self.attn_mqa(
                q_nope_out, k_nope, k_nope, forward_batch, q_rope=q_pe, k_rope=k_rope, kv_b_proj = self.kv_b_proj
            )    
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([torch.repeat_interleave(k_nope, repeats=self.num_local_heads, dim=1), torch.repeat_interleave(k_pe, repeats= (self.num_local_heads // self.num_kv_heads), dim=1)], dim=-1)
            attn_output = self.attn_mqa(q, k, k_nope, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = attn_bmm_output[:, :expected_m, :]
        elif self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            if self.w_fused_T is not None:
                pass
            else:
                assert self.w_vc.shape[0] == 1
                w_vc_per_head = self.w_vc.view(self.w_vc.shape[1], self.num_kv_heads, -1)
                w_vc_per_head = torch.repeat_interleave(w_vc_per_head, repeats= (self.num_local_heads // self.num_kv_heads), dim=-2)
                attn_bmm_output = torch.einsum("l h d, b h l -> b h d", w_vc_per_head, attn_output)

        if self.w_fused_T is not None:
            output    = torch.mm(attn_output.flatten(1), self.w_fused_T)   # (B, hidden)
        else:
            output = attn_bmm_output.view(attn_bmm_output.shape[0], -1)
            output, _ = self.o_proj(output)


        return output

    def forward_absorb_fused_mla_rope_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim_per_head
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim_per_head, self.qk_rope_head_dim_per_head], dim=-1)

        if self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        return (
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            enable_rope_fusion,
            k_input,
            forward_batch,
            zero_allocator,
        )

    def forward_absorb_fused_mla_rope_core(
        self,
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
    ):
        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            # Fetch latent cache from memory pool with precomputed chunked kv indices
            latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[
                forward_batch.prefix_chunk_kv_indices[i]
            ].contiguous()

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            lse = torch.transpose(lse, 0, 1).contiguous()
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse

        return accum_output

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            all_q = self.q_proj(hidden_states)[0]
            all_q_nope, all_q_pe = all_q.split([self.num_local_heads * self.qk_nope_head_dim_per_head, self.num_local_heads * self.qk_rope_head_dim_per_head], dim=-1)
            all_q_nope = all_q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim_per_head)
            all_q_pe = all_q_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim_per_head)
            q = torch.cat([all_q_nope, all_q_pe], dim=-1)
            # q = self.q_proj(hidden_states)[0].view(
            #     -1, self.num_local_heads, self.qk_head_dim
            # )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )

        return q, k, v, forward_batch

    def forward_normal_chunked_kv_core(self, q, k, v, forward_batch):
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        lse = torch.transpose(lse, 0, 1).contiguous()

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if any(forward_batch.extend_prefix_lens_cpu):
            # Only initialize the info once
            if forward_batch.num_prefix_chunks is None:
                forward_batch.prepare_chunked_prefix_cache_info(q.device)

            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


EntryClass = [LlamaForCausalLMMLA]
