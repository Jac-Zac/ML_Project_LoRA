# class MultiheadAttentionLoRA(LoRA[nn.MultiheadAttention]):
#     """
#     NOTE: MultiheadAttention doesn't quite fit the "sidecar" pattern, like essentially
#     every other module does.  Unlike other modules, which typically have a single
#     'weight' parameter that is modified by LoRA, MultiheadAttention has multiple
#     parameters that are modified by LoRA, and those parameters interact in non-trivial
#     ways (via attention) within the module itself.
#
#     For that reason, we emulate all of the necessary properties of MultiheadAttention,
#     and reuse the 'forward' method from MultiheadAttention.  This allows us to
#     dynamically compute the LoRA-adjusted parameters without rewriting *all* of the
#     logic from 'MultiheadAttention.forward'.
#     """
#
#     def __init__(
#         self,
#         module: nn.MultiheadAttention,
#         lora_module: Optional[MultiheadAttentionLoRAModule],
#         enabled: bool = True,
#     ):
#         super().__init__(module, lora_module, enabled=enabled)
#         self.module = cast(nn.MultiheadAttention, self.module)
#         self.lora_module = cast(
#             Optional[MultiheadAttentionLoRAModule], self.lora_module
#         )
#
#     def forward(  # type: ignore
#         self,
#         query: Tensor,
#         key: Tensor,
#         value: Tensor,
#         key_padding_mask: Optional[Tensor] = None,
#         need_weights: bool = True,
#         attn_mask: Optional[Tensor] = None,
#         average_attn_weights: bool = True,
#         is_causal: bool = False,
#     ) -> Tuple[Tensor, Optional[Tensor]]:
#         if (not self.enabled) or self.lora_module is None:
#             return self.module.forward(
#                 query,
#                 key,
#                 value,
#                 key_padding_mask=key_padding_mask,
#                 need_weights=need_weights,
#                 attn_mask=attn_mask,
#                 average_attn_weights=average_attn_weights,
#                 is_causal=is_causal,
#             )
#         return nn.MultiheadAttention.forward(
#             cast(nn.MultiheadAttention, self),
#             query,
#             key,
#             value,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#             attn_mask=attn_mask,
#             average_attn_weights=average_attn_weights,
#             is_causal=is_causal,
#         )
#
#     def merge_masks(
#         self,
#         attn_mask: Optional[Tensor],
#         key_padding_mask: Optional[Tensor],
#         query: Tensor,
#     ) -> Tuple[Optional[Tensor], Optional[int]]:
#         return nn.MultiheadAttention.merge_masks(
#             cast(nn.MultiheadAttention, self),
#             attn_mask,
#             key_padding_mask,
#             query,
#         )
#
#     @property
#     def embed_dim(self) -> int:
#         return self.module.embed_dim
#
#     @property
#     def num_heads(self) -> int:
#         return self.module.num_heads
#
#     @property
#     def dropout(self) -> float:
#         return self.module.dropout
#
#     @property
#     def add_zero_attn(self) -> bool:
#         return self.module.add_zero_attn
#
#     @property
#     def batch_first(self) -> bool:
#         return self.module.batch_first
#
#     @property
#     def _qkv_same_embed_dim(self) -> bool:
#         return self.module._qkv_same_embed_dim
#
#     @property
#     def bias_k(self) -> Optional[Tensor]:
#         if self.module.bias_k is None:
#             return None
#         return self.module.bias_k.data.detach()
#
#     @property
#     def bias_v(self) -> Optional[Tensor]:
#         if self.module.bias_v is None:
#             return None
#         return self.module.bias_v.data.detach()
#
#     @property
#     def in_proj_weight(self) -> Tensor:
#         in_proj_weight = self.module.in_proj_weight
#         if in_proj_weight is None:
#             return None
#
#         weight = in_proj_weight.data.detach()
#         if (
#             self.enabled
#             and self.lora_module is not None
#             and self.lora_module.in_proj_weight is not None
#         ):
#             return weight + self.lora_module.in_proj_weight
#         else:
#             return weight
#
#     @property
#     def in_proj_bias(self) -> Tensor:
#         bias = self.module.in_proj_bias
#         if bias is None:
#             return None
#         else:
#             return bias.data.detach()
#         # TODO: Add support for 'in_proj_bias' in MultiheadAttentionLoRAModule
#
#     @property
#     def q_proj_weight(self) -> Optional[Tensor]:
#         weight = self.module.q_proj_weight.data.detach()
#         if self.enabled and weight is not None and self.lora_module is not None:
#             return weight + self.lora_module.q_proj_weight
#         else:
#             return weight
#
#     @property
#     def k_proj_weight(self) -> Optional[Tensor]:
#         weight = self.module.k_proj_weight.data.detach()
#         if self.enabled and (weight is not None) and (self.lora_module is not None):
#             return weight + self.lora_module.k_proj_weight
#         else:
#             return weight
#
#     @property
#     def v_proj_weight(self) -> Optional[Tensor]:
#         weight = self.module.v_proj_weight.data.detach()
#         if self.enabled and (weight is not None) and (self.lora_module is not None):
#             return weight + self.lora_module.v_proj_weight
#         else:
#             return weight
#
#     @property
#     def out_proj(self) -> OutProj:
#         weight = self.module.out_proj.weight.data.detach()
#         bias = self.module.out_proj.bias
#         if self.enabled and self.lora_module is not None:
#             lora_out_proj = cast(OutProj, self.lora_module.out_proj)
#             weight = weight + lora_out_proj.weight
#             if (bias is not None) and (lora_out_proj.bias is not None):
#                 # Mypy complains about a type mismatch here (Tensor vs. Parameter)
#                 # but Parameter is just a subclass of Tensor, so this is fine.
#                 bias = bias + lora_out_proj.bias  # type: ignore
#
#         return OutProj(weight, bias)
#
#
# class OutProj(NamedTuple):
#     weight: Tensor
#     bias: Optional[Tensor]
