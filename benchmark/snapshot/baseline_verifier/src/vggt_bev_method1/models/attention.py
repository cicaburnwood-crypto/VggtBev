from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class MultiheadAttention(nn.Module):
    """Exact or global linearized multi-head attention.

    Linear mode preserves all-query/all-token connectivity without materializing
    the quadratic attention matrix, making one query per 512x512 BEV cell viable.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        *,
        mode: str = "linear",
        projection_fusion: str = "separate",
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if mode not in ("linear", "exact"):
            raise ValueError("attention mode must be 'linear' or 'exact'")
        if projection_fusion not in ("separate", "qkv", "kv"):
            raise ValueError("projection_fusion must be separate, qkv, or kv")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.mode = mode
        # Execution-only choice: all Parameter objects and state_dict keys stay
        # unchanged. Concatenating their small weights lets the 640K-query
        # linear-attention path issue one GEMM instead of three for self
        # attention, and one instead of two for cross-attention K/V.
        self.projection_fusion = projection_fusion
        # Execution-only callable; absent from state_dict/checkpoint contracts.
        self._compiled_forward_impl = None
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def configure_execution_compilation(
        self,
        *,
        backend: str,
        mode: str,
        dynamic: bool,
    ) -> bool:
        """Compile dense linear attention while leaving exact scale attention eager."""

        if self.mode != "linear":
            return False
        self._compiled_forward_impl = torch.compile(
            self._forward_impl,
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            fullgraph=False,
        )
        return True

    def _forward_impl(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.projection_fusion == "qkv":
            if query.shape != context.shape:
                raise ValueError("QKV fusion requires matching query/context shapes")
            weight = torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0
            )
            bias = torch.cat(
                (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), dim=0
            )
            q_raw, k_raw, v_raw = F.linear(query, weight, bias).split(
                self.hidden_dim, dim=-1
            )
        elif self.projection_fusion == "kv":
            q_raw = self.q_proj(query)
            weight = torch.cat((self.k_proj.weight, self.v_proj.weight), dim=0)
            bias = torch.cat((self.k_proj.bias, self.v_proj.bias), dim=0)
            k_raw, v_raw = F.linear(context, weight, bias).split(
                self.hidden_dim, dim=-1
            )
        else:
            q_raw = self.q_proj(query)
            k_raw = self.k_proj(context)
            v_raw = self.v_proj(context)
        q = self._split(q_raw)
        k = self._split(k_raw)
        v = self._split(v_raw)
        if context_weight is not None:
            if context_weight.shape != context.shape[:2]:
                raise ValueError(
                    "context_weight must have shape [batch, context_tokens]"
                )
            nonnegative = ~(context_weight < 0).any()
            if nonnegative.device.type == "cuda":
                torch._assert_async(
                    nonnegative,
                    "context attention weights cannot be negative",
                )
            elif not bool(nonnegative):
                raise ValueError("context attention weights cannot be negative")
            context_weight = context_weight.to(device=k.device, dtype=k.dtype)
        if self.mode == "exact":
            attention_bias = None
            if context_weight is not None:
                attention_bias = context_weight.clamp_min(1e-12).log()[
                    :, None, None, :
                ]
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_bias,
            )
        else:
            scale = self.head_dim**-0.25
            q_feature = F.elu(q * scale) + 1.0
            k_feature = F.elu(k * scale) + 1.0
            if context_weight is not None:
                k_feature = k_feature * context_weight[:, None, :, None]
            key_value = torch.einsum("bhmd,bhme->bhde", k_feature, v)
            key_sum = k_feature.sum(dim=2)
            denominator = torch.einsum(
                "bhqd,bhd->bhq", q_feature, key_sum
            ).clamp_min(1e-6)
            output = torch.einsum(
                "bhqd,bhde,bhq->bhqe",
                q_feature,
                key_value,
                denominator.reciprocal(),
            )
        output = output.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.hidden_dim
        )
        return self.out_proj(output)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        implementation = self._compiled_forward_impl or self._forward_impl
        if context_weight is None:
            return implementation(query, context)
        return implementation(query, context, context_weight)


class DeformableCrossAttention(nn.Module):
    """PDF Method I deformable cross-attention over spatial token pyramids.

    Each BEV query predicts a small set of image sampling locations. Sampling
    is performed at every VGGT feature level and every input frame, then
    aggregated without constructing a Q-by-image-token attention matrix.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        *,
        levels: int,
        samples: int,
        query_chunk_size: int,
        maximum_offset: float = 0.75,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if levels <= 0 or samples <= 0 or query_chunk_size <= 0:
            raise ValueError("levels, samples, and query_chunk_size must be positive")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.levels = levels
        self.samples = samples
        self.query_chunk_size = query_chunk_size
        self.maximum_offset = maximum_offset
        # Execution-only switch. It is intentionally absent from configs and
        # state_dicts because it does not change the model or checkpoint
        # contract.
        self.memory_efficient_training = True
        self.memory_efficient_checkpoint_fraction = 1.0
        self._compiled_forward_query_chunk = None
        self._compiled_project_key_value = None
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projections = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(levels)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(levels)]
        )
        self.query_offset_projection = nn.Linear(
            hidden_dim,
            levels * samples * 2,
        )
        self.frame_offset_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, samples * 2) for _ in range(levels)]
        )
        self.attention_bias_projection = nn.Linear(
            hidden_dim,
            heads * levels * samples,
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def configure_query_chunk_compilation(
        self,
        *,
        backend: str,
        mode: str,
        dynamic: bool,
    ) -> None:
        """Compile only the bounded deformable-attention hot kernel."""

        self._compiled_forward_query_chunk = torch.compile(
            self._forward_query_chunk,
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            fullgraph=False,
        )
        self._compiled_project_key_value = torch.compile(
            self._project_key_value,
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            fullgraph=False,
        )

    @staticmethod
    def _project_key_value(
        flattened: torch.Tensor,
        key_weight: torch.Tensor,
        key_bias: torch.Tensor | None,
        value_weight: torch.Tensor,
        value_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply equivalent K/V 1x1 projections in one convolution launch.

        Weights remain separate Parameters, so state_dict keys, optimizer state,
        gradients, and checkpoint compatibility are unchanged.
        """

        if (key_bias is None) != (value_bias is None):
            raise ValueError("key and value projections must use matching bias modes")
        weight = torch.cat((key_weight, value_weight), dim=0)
        bias = (
            None
            if key_bias is None
            else torch.cat((key_bias, value_bias), dim=0)
        )
        return F.conv2d(flattened, weight, bias)

    def _forward_query_chunk(
        self,
        query_chunk: torch.Tensor,
        reference_chunk: torch.Tensor,
        *projected_level_tensors: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one query chunk.

        Keeping this boundary inside deformable attention lets training
        rematerialize only the sampled key/value and softmax intermediates.
        VGGT, token projection, self-attention, FFN, and losses are not
        recomputed.
        """

        if len(projected_level_tensors) != self.levels * 2:
            raise ValueError("each feature level must provide key/value and offsets")
        projected_levels = [
            (
                projected_level_tensors[2 * index],
                projected_level_tensors[2 * index + 1],
            )
            for index in range(self.levels)
        ]
        batch, chunk_size, _ = query_chunk.shape
        query_heads = self.query_projection(query_chunk).view(
            batch,
            chunk_size,
            self.heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        query_offsets = self.query_offset_projection(query_chunk).view(
            batch,
            chunk_size,
            self.levels,
            self.samples,
            2,
        )
        query_offsets = torch.tanh(query_offsets) * self.maximum_offset
        attention_bias = self.attention_bias_projection(query_chunk).view(
            batch,
            chunk_size,
            self.heads,
            self.levels,
            self.samples,
        ).permute(0, 2, 1, 3, 4)
        running_max = query_chunk.new_full(
            (batch, self.heads, chunk_size),
            -torch.inf,
        )
        denominator = query_chunk.new_zeros(batch, self.heads, chunk_size)
        numerator = query_chunk.new_zeros(
            batch,
            self.heads,
            chunk_size,
            self.head_dim,
        )
        for level_index, (
            key_value_feature,
            frame_offsets,
        ) in enumerate(projected_levels):
            _, frames, _, height, width = key_value_feature.shape
            grid = (
                reference_chunk[None, :, None, None, :]
                + query_offsets[:, :, level_index, None]
                + frame_offsets[:, None]
            ).clamp(-1.0, 1.0)
            grid = (
                grid.permute(0, 2, 1, 3, 4)
                .reshape(batch * frames, chunk_size, self.samples, 2)
            )
            # Bilinear sampling is channel-independent, so concatenating K/V
            # is exactly equivalent to two calls while halving grid_sample
            # launches and grid reads.
            sampled_key_value = F.grid_sample(
                key_value_feature.reshape(
                    batch * frames,
                    self.hidden_dim * 2,
                    height,
                    width,
                ),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_key_value = sampled_key_value.view(
                batch,
                frames,
                2,
                self.heads,
                self.head_dim,
                chunk_size,
                self.samples,
            )
            sampled_key = sampled_key_value[:, :, 0].permute(0, 2, 4, 1, 5, 3)
            sampled_value = sampled_key_value[:, :, 1].permute(0, 2, 4, 1, 5, 3)
            scores = (
                query_heads[:, :, :, None, None] * sampled_key
            ).sum(dim=-1) * (self.head_dim**-0.5)
            scores = scores + attention_bias[
                :, :, :, level_index, :
            ][:, :, :, None, :]
            level_max = scores.amax(dim=(3, 4))
            new_max = torch.maximum(running_max, level_max)
            previous_scale = torch.exp(running_max - new_max)
            weights = torch.exp(
                scores - new_max[:, :, :, None, None]
            )
            numerator = (
                numerator * previous_scale[..., None]
                + (weights[..., None] * sampled_value).sum(dim=(3, 4))
            )
            denominator = (
                denominator * previous_scale + weights.sum(dim=(3, 4))
            )
            running_max = new_max
        normalized = numerator / denominator.clamp_min(1e-6)[..., None]
        output = normalized.transpose(1, 2).reshape(
            batch,
            chunk_size,
            self.hidden_dim,
        )
        return self.output_projection(output)

    def forward(
        self,
        query: torch.Tensor,
        pyramid: list[torch.Tensor],
        reference_grid: torch.Tensor,
    ) -> torch.Tensor:
        if len(pyramid) != self.levels:
            raise ValueError(
                f"expected {self.levels} pyramid levels, got {len(pyramid)}"
            )
        batch, query_count, _ = query.shape
        if reference_grid.shape != (query_count, 2):
            raise ValueError("reference_grid must have shape [Q, 2]")

        projected_levels = []
        frame_counts = set()
        for level_index, feature in enumerate(pyramid):
            if feature.ndim != 5 or feature.shape[0] != batch:
                raise ValueError("pyramid levels must have shape [B, N, C, H, W]")
            level_batch, frames, channels, height, width = feature.shape
            if channels != self.hidden_dim:
                raise ValueError("pyramid channel width does not match hidden_dim")
            frame_counts.add(frames)
            flattened = feature.reshape(
                level_batch * frames,
                channels,
                height,
                width,
            )
            key_projection = self.key_projections[level_index]
            value_projection = self.value_projections[level_index]
            projection = (
                self._compiled_project_key_value
                if self._compiled_project_key_value is not None
                else self._project_key_value
            )
            key_value = projection(
                flattened,
                key_projection.weight,
                key_projection.bias,
                value_projection.weight,
                value_projection.bias,
            )
            frame_summary = feature.mean(dim=(-2, -1))
            frame_offsets = self.frame_offset_projections[level_index](
                frame_summary
            ).view(batch, frames, self.samples, 2)
            frame_offsets = torch.tanh(frame_offsets) * (self.maximum_offset * 0.5)
            projected_levels.append(
                (
                    key_value.view(
                        level_batch,
                        frames,
                        channels * 2,
                        height,
                        width,
                    ),
                    frame_offsets,
                )
            )
        if len(frame_counts) != 1:
            raise ValueError("all pyramid levels must contain the same frame count")

        outputs = []
        reference_grid = reference_grid.to(device=query.device, dtype=query.dtype)
        flat_projected_levels = tuple(
            tensor
            for projected_level in projected_levels
            for tensor in projected_level
        )
        query_chunk_forward = (
            self._compiled_forward_query_chunk
            if self._compiled_forward_query_chunk is not None
            else self._forward_query_chunk
        )
        chunk_count = (
            query_count + self.query_chunk_size - 1
        ) // self.query_chunk_size
        checkpoint_fraction = min(
            max(float(self.memory_efficient_checkpoint_fraction), 0.0),
            1.0,
        )
        checkpoint_chunk_count = min(
            chunk_count,
            int(chunk_count * checkpoint_fraction + 0.999999),
        )
        for chunk_index, start in enumerate(
            range(0, query_count, self.query_chunk_size)
        ):
            end = min(start + self.query_chunk_size, query_count)
            query_chunk = query[:, start:end]
            reference_chunk = reference_grid[start:end]
            requires_backward = query_chunk.requires_grad or any(
                tensor.requires_grad for tensor in flat_projected_levels
            )
            if (
                self.memory_efficient_training
                and chunk_index < checkpoint_chunk_count
                and torch.is_grad_enabled()
                and requires_backward
            ):
                outputs.append(
                    checkpoint(
                        query_chunk_forward,
                        query_chunk,
                        reference_chunk,
                        *flat_projected_levels,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                )
            else:
                outputs.append(
                    query_chunk_forward(
                        query_chunk,
                        reference_chunk,
                        *flat_projected_levels,
                    )
                )
        return torch.cat(outputs, dim=1)


class DirectDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        *,
        self_attention_mode: str,
        cross_attention_mode: str,
        feature_levels: int,
        deformable_samples: int,
        cross_query_chunk_size: int,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if cross_attention_mode not in ("linear", "exact", "deformable"):
            raise ValueError(
                "cross_attention_mode must be linear, exact, or deformable"
            )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiheadAttention(
            hidden_dim,
            heads,
            mode=self_attention_mode,
            projection_fusion=(
                "qkv" if self_attention_mode == "linear" else "separate"
            ),
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention_mode = cross_attention_mode
        self.cross_attention = (
            DeformableCrossAttention(
                hidden_dim,
                heads,
                levels=feature_levels,
                samples=deformable_samples,
                query_chunk_size=cross_query_chunk_size,
            )
            if cross_attention_mode == "deformable"
            else MultiheadAttention(
                hidden_dim,
                heads,
                mode=cross_attention_mode,
                projection_fusion=(
                    "kv" if cross_attention_mode == "linear" else "separate"
                ),
            )
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )
        # Compile only the dense post-cross residual, not the Python chunk loop.
        self._compiled_ffn_residual = None

    def configure_execution_compilation(
        self,
        *,
        backend: str,
        mode: str,
        dynamic: bool,
    ) -> bool:
        self._compiled_ffn_residual = torch.compile(
            self._ffn_residual,
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            fullgraph=False,
        )
        return True

    def _ffn_residual(self, query: torch.Tensor) -> torch.Tensor:
        return query + self.ffn(self.ffn_norm(query))

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | list[torch.Tensor],
        reference_grid: torch.Tensor,
        context_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.self_norm(query)
        query = query + self.self_attention(normalized, normalized)
        normalized_query = self.cross_norm(query)
        if self.cross_attention_mode == "deformable":
            if context_weight is not None:
                raise ValueError(
                    "context_weight is supported only by global attention"
                )
            if not isinstance(context, list):
                raise TypeError("deformable cross-attention requires a feature pyramid")
            normalized_context = [
                self.context_norm(level.movedim(2, -1)).movedim(-1, 2)
                for level in context
            ]
            cross = self.cross_attention(
                normalized_query,
                normalized_context,
                reference_grid,
            )
        else:
            if not isinstance(context, torch.Tensor):
                raise TypeError("global cross-attention requires a flat context tensor")
            cross = self.cross_attention(
                normalized_query,
                self.context_norm(context),
                context_weight,
            )
        query = query + cross
        implementation = self._compiled_ffn_residual or self._ffn_residual
        return implementation(query)
