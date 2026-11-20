# FeaXDrive2 release: extracted or reorganized from the research implementation.
# Copyright 2025 The Xiaomi Corporation. All rights reserved.
#
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
"""Flow-only extraction of the evaluated planner; module/state names are retained."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F
from timm.layers import Mlp
from transformers.feature_extraction_utils import BatchFeature
from .blocks.encoder import ActionEncoder, StateAttentionEncoder
from .feaxdrive_dit import LightningDiT

@dataclass
class FlowConfig:
    time_sampling: str = "uniform"
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    mean_variance_net: bool = False

@dataclass
class FlowPlannerConfig:
    diffusion_model_cfg: dict = field(default_factory=lambda: dict(
        num_heads=8, head_dim=48, num_layers=16, output_dim=512,
        dropout=0., attention_bias=True, norm_eps=1e-5, interleave_attention=True,
        semantic_fusion_mode="sparse_interleaved", sparse_cross_attention_layers=list(range(1,16,2))))
    input_embedding_dim: int = 384
    hidden_size: int = 1024
    action_dim: int = 3
    action_horizon: int = 8
    add_pos_embed: bool = True
    max_seq_len: int = 8
    ego_status_encoder_type: str = "mlp"
    use_condition_mean_fusion: bool = False
    sampling_method: str = "flow"
    num_inference_steps: int = 5
    vlm_size: str = "small"
    record_delta_pi_curve: bool = False
    flow_cfg: FlowConfig = field(default_factory=FlowConfig)

    def __post_init__(self):
        if self.sampling_method != "flow" or self.num_inference_steps != 5:
            raise ValueError("This release implements the paper's five-step Flow planner")


class FlowPlanner(nn.Module):

    def __init__(self, config: FlowPlannerConfig):
        super().__init__()
        self.config = config
        self.model = LightningDiT(**config.diffusion_model_cfg)
        self.his_traj_encoder = Mlp(in_features=12, hidden_features=config.hidden_size, out_features=config.input_embedding_dim, norm_layer=nn.LayerNorm)
        if config.ego_status_encoder_type == 'attention':
            self.ego_status_encoder = StateAttentionEncoder(state_dim=8, embed_dim=config.input_embedding_dim, num_kinematic_states=4)
        else:
            self.ego_status_encoder = Mlp(in_features=8, hidden_features=config.hidden_size, out_features=config.input_embedding_dim, norm_layer=nn.LayerNorm)
        self.action_encoder = ActionEncoder(action_dim=config.action_dim, hidden_size=config.input_embedding_dim)
        if config.vlm_size == 'large':
            self.feature_encoder = nn.Linear(3584, config.input_embedding_dim)
        else:
            self.feature_encoder = nn.Linear(1536, config.input_embedding_dim)
        fusion_inputs = 3 if bool(config.use_condition_mean_fusion) else 2
        self.fusion_projector = nn.Linear(config.input_embedding_dim * fusion_inputs, config.input_embedding_dim)
        output_dim = 2 * config.action_dim if config.sampling_method == 'flow' and config.flow_cfg.mean_variance_net else config.action_dim
        self.action_decoder = Mlp(in_features=self.model.output_dim, hidden_features=config.hidden_size, out_features=output_dim, norm_layer=nn.LayerNorm)
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self._init_flow_sampler(config.flow_cfg)

    def _prepare_status_feature(self, status_feature: torch.Tensor) -> torch.Tensor:
        """Apply a planner-specific status policy before AdaLN conditioning."""
        return status_feature

    def _fuse_trajectory_inputs(self, history_features: torch.Tensor, condition_features: torch.Tensor, action_features: torch.Tensor, condition_attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        parts = [history_features]
        if bool(self.config.use_condition_mean_fusion):
            if condition_attention_mask is None:
                condition_mean = condition_features.mean(1)
            else:
                weights = condition_attention_mask.unsqueeze(-1).to(dtype=condition_features.dtype, device=condition_features.device)
                condition_mean = (condition_features * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            condition_mean = condition_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            parts.append(condition_mean)
        parts.append(action_features)
        return self.fusion_projector(torch.cat(parts, dim=2))

    def _init_flow_sampler(self, cfg: FlowConfig):
        """Initializes components required for Flow Matching."""
        self.beta_dist = Beta(cfg.noise_beta_alpha, cfg.noise_beta_beta)
        self.num_timestep_buckets = cfg.num_timestep_buckets

    def sample_time(self, batch_size, device, dtype):
        """Samples time for training based on the sampling method.

        Flow Matching baseline uses a clean CondOT path with uniform t in [0, 1).
        The previous beta schedule is kept as an explicit ablation via
        config.flow_cfg.time_sampling="beta".
        """
        mode = str(getattr(self.config.flow_cfg, 'time_sampling', 'uniform')).lower()
        if mode == 'uniform':
            t = torch.rand(batch_size, device=device, dtype=dtype)
        elif mode == 'beta':
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            t = (self.config.flow_cfg.noise_s - sample) / self.config.flow_cfg.noise_s
        else:
            raise ValueError(f'Unsupported flow time_sampling: {mode!r}')
        return t.clamp(0.0, 1.0 - 1e-05)

    def flow_time_to_bucket(self, t_cont: torch.Tensor) -> torch.Tensor:
        """Maps continuous flow time in [0, 1) to integer DiT time buckets."""
        n = int(getattr(self.config.flow_cfg, 'num_timestep_buckets', 1000))
        return torch.clamp((t_cont * n).long(), min=0, max=n - 1)

    def forward(self, vl_features: torch.Tensor, action_input: BatchFeature) -> BatchFeature:
        """
        Computes the training loss for a given batch.

        Args:
            vl_features (torch.Tensor): The vision-language features from the backbone.
            action_input (BatchFeature): A batch containing ground truth actions and other conditioning.

        Returns:
            BatchFeature: A batch containing the computed loss.
        """
        if any((key.startswith(('flow_rcf_', 'flow_local_')) for key in action_input)):
            from ..training.rcf import forward_loss
            return forward_loss(self, vl_features, action_input)
        vl_embeds, encoder_attention_mask = self._prepare_vl_conditioning(vl_features, action_input)
        his_traj_features = self.his_traj_encoder(action_input.his_traj.unsqueeze(1)).repeat(1, self.config.action_horizon, 1)
        ego_status_features = self.ego_status_encoder(self._prepare_status_feature(action_input.status_feature))
        gt_actions = self.norm_odo(action_input.action)
        noise = action_input.get('flow_noise')
        if noise is None:
            noise = torch.randn_like(gt_actions)
        t_cont = action_input.get('flow_time')
        if t_cont is None:
            t_cont = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
        if noise.shape != gt_actions.shape or t_cont.shape != gt_actions.shape[:1]:
            raise ValueError('Flow noise/time shape mismatch')
        if 'flow_noise' in action_input and (not torch.isfinite(noise).all()):
            raise ValueError('Nonfinite supplied Flow noise')
        if 'flow_time' in action_input and (not torch.isfinite(t_cont).all() or ((t_cont < 0) | (t_cont >= 1)).any()):
            raise ValueError('Supplied Flow time must be finite in [0,1)')
        camera_context = None
        if hasattr(self.model, 'camera_grounded_query'):
            from .camera_grounded_flow import discrete_flow_time, rollout_prefix
            t_cont = discrete_flow_time(t_cont)
            previous = rollout_prefix(noise, t_cont, lambda x, t, p: self.predict_flow_velocity(x, t, vl_embeds, his_traj_features, ego_status_features, encoder_attention_mask, action_input.get('vehicle_geometry'), camera_inputs=action_input, previous_metric=p), self.denorm_odo)
            camera_context = self._camera_query_context(action_input, previous, t_cont)
        t_cont_reshaped = t_cont[:, None, None]
        noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
        velocity_target = gt_actions - noise
        t_discrete = self.flow_time_to_bucket(t_cont)
        action_features = self.action_encoder(noisy_actions, t_discrete)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
            action_features += self.position_embedding(pos_ids)
        fused_input = self._fuse_trajectory_inputs(his_traj_features, vl_embeds, action_features, condition_attention_mask=encoder_attention_mask)
        model_output = self.model(fused_input, vl_embeds, ego_status_features, t_discrete, encoder_attention_mask=encoder_attention_mask, **{'vehicle_geometry': action_input.get('vehicle_geometry')} if any((hasattr(self.model, n) for n in ('geometry_conditioner', 'footprint_query', 'camera_grounded_query'))) else {}, **{'camera_context': camera_context} if camera_context is not None else {})
        pred = self.action_decoder(model_output)
        pred_velocity = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
        loss_flow = F.mse_loss(pred_velocity, velocity_target, reduction='mean')
        weights = action_input.get('flow_loss_weights')
        if weights is not None:
            if weights.shape != gt_actions.shape[:1] or not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Invalid per-sample Flow loss weights')
            per_sample = F.mse_loss(pred_velocity.float(), velocity_target.float(), reduction='none').mean((1, 2))
            loss_flow = (per_sample * weights).sum()
        flow_output = {'loss': loss_flow, 'loss_base': loss_flow.detach(), 'loss_flow': loss_flow.detach(), 'flow_t_mean': t_cont.mean().detach()}
        if any((k in action_input for k in ('flow_reference_velocity', 'flow_reference_actions', 'flow_reference_time', 'flow_preservation_weights'))):
            if 'g3_pair_weights' in action_input or not hasattr(self.model, 'camera_grounded_query'):
                raise ValueError('E-Body preservation requires Camera FM without preference loss')
            from ..training.preservation import preservation_loss
            preserve = preservation_loss(pred_velocity, noisy_actions, t_cont, action_input)
            flow_output.update(loss=loss_flow + preserve, loss_preserve=preserve.detach())
        return BatchFeature(data=flow_output)

    @staticmethod
    def _camera_query_context(inputs, previous_metric, flow_time, *, retain_trajectory_grad=False):
        from .camera_contract import CAMERA_KEYS
        if inputs is None or previous_metric is None:
            raise ValueError('Missing Camera Query spatial input or causal prefix')
        result = {k: inputs[k] for k in CAMERA_KEYS}
        result.update(trajectory=previous_metric if retain_trajectory_grad else previous_metric.detach(), flow_time=flow_time)
        if 'projection_geometry' in inputs:
            result['projection_geometry'] = inputs['projection_geometry']
        return result

    def predict_flow_velocity(self, actions, t_cont, vl_embeds, history_embeds, ego_embeds, encoder_attention_mask=None, vehicle_geometry=None, camera_inputs=None, previous_metric=None):
        """Shared canonical velocity field for Euler inference and opt-in Flow RL."""
        t = self.flow_time_to_bucket(t_cont)
        action_features = self.action_encoder(actions, t)
        if hasattr(self, 'position_embedding'):
            action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=actions.device))
        fused_input = self._fuse_trajectory_inputs(history_embeds, vl_embeds, action_features, condition_attention_mask=encoder_attention_mask)
        model_output = self.model(fused_input, vl_embeds, ego_embeds, t, encoder_attention_mask=encoder_attention_mask, **{'vehicle_geometry': vehicle_geometry} if any((hasattr(self.model, n) for n in ('geometry_conditioner', 'footprint_query', 'camera_grounded_query'))) else {}, **{'camera_context': self._camera_query_context(camera_inputs, previous_metric, t_cont, **{'retain_trajectory_grad': True} if getattr(self.model.camera_grounded_query, 'retain_trajectory_grad', False) else {})} if hasattr(self.model, 'camera_grounded_query') else {})
        pred = self.action_decoder(model_output)
        return pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred

    def get_action(self, vl_features: torch.Tensor, action_input: BatchFeature, init_actions: Optional[torch.Tensor]=None, deterministic: bool=False, *, return_trace: bool=False, prepared_condition=None, query_prefixes: Optional[torch.Tensor]=None, query_update: str='dynamic') -> BatchFeature:
        """
        Generates action trajectories via the configured sampling method.

        This method preserves the five-step Euler update and clipping order.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            action_input (BatchFeature): Input containing conditioning features like
                historical trajectory and ego status.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start
                the denoising from. If None, starts from pure noise.
            deterministic (bool): Retained call signature; Flow noise is supplied explicitly.
            return_trace (bool): Opt-in Flow trace for training or analysis; unclipped states
                and clean estimates retain gradients, Query prefixes do not.
            prepared_condition: Internal reuse of the same observation encodings
                (vl, mask, history, status); never a teacher/state override.
            query_prefixes: Training-only detached metric Query coordinates
                [B,steps,horizon,3]. Requires Camera Query and return_trace;
                substitutes no action states. None preserves inference exactly.
            query_update: "dynamic" preserves the canonical sampler. Eval-only
                "fixed_first_estimate" requires five-step Camera Flow, eval()
                and disabled autograd, and excludes query_prefixes. Start with
                literal metric zeros; after velocity evaluation 0, clamp its
                pre-Euler clean estimate, denormalize and detach it as r1. Reuse
                r1 for evaluations 1..4, updating latent/time/context normally.
                The first velocity that can differ from dynamic is evaluation 2.

        Returns:
            BatchFeature: A batch containing the final predicted trajectory.
        """
        if query_update not in ('dynamic', 'fixed_first_estimate'):
            raise ValueError(f'Unsupported query_update: {query_update!r}')
        fixed_first_estimate = query_update == 'fixed_first_estimate'
        if fixed_first_estimate:
            if self.config.sampling_method != 'flow' or self.config.num_inference_steps != 5 or (not hasattr(self.model, 'camera_grounded_query')):
                raise ValueError('fixed_first_estimate requires five-step Camera Flow')
            if self.training or torch.is_grad_enabled():
                raise ValueError('fixed_first_estimate is eval-only; call eval() with autograd disabled')
            if query_prefixes is not None:
                raise ValueError('fixed_first_estimate cannot be combined with training query_prefixes')
        if return_trace and self.config.sampling_method != 'flow':
            raise ValueError('A differentiable trace is available only for canonical Flow')
        if query_prefixes is not None:
            if not torch.is_grad_enabled() or getattr(self, 'rcf_mechanism', 'full') != 'reference_prefix' or (not return_trace) or (self.config.sampling_method != 'flow') or (not hasattr(self.model, 'camera_grounded_query')) or (not isinstance(query_prefixes, torch.Tensor)) or (query_prefixes.shape != (len(vl_features), self.config.num_inference_steps, self.config.action_horizon, self.config.action_dim)) or (not query_prefixes.is_floating_point()) or query_prefixes.requires_grad or (query_prefixes.device != vl_features.device) or (not torch.isfinite(query_prefixes).all()) or torch.count_nonzero(query_prefixes[:, 0]):
                raise ValueError('query_prefixes requires training Camera Flow trace, detached finite [B,steps,horizon,3] and literal metric-zero first prefix')
        if prepared_condition is None:
            vl_embeds, encoder_attention_mask = self._prepare_vl_conditioning(vl_features, action_input)
        else:
            vl_embeds, encoder_attention_mask, history_embeds, ego_embeds = prepared_condition
        if bool(getattr(self.config, 'record_delta_pi_curve', False)):
            self._delta_pi_curve = []
        if prepared_condition is None:
            history_embeds = self.his_traj_encoder(action_input.his_traj.unsqueeze(1)).repeat(1, self.config.action_horizon, 1)
            ego_embeds = self.ego_status_encoder(self._prepare_status_feature(action_input.status_feature))
        B, D = (vl_embeds.shape[0], self.config.action_dim)
        device, dtype = (vl_embeds.device, vl_embeds.dtype)
        current_actions = init_actions if init_actions is not None else torch.randn((B, self.config.action_horizon, D), device=device, dtype=dtype)
        num_steps = int(self.config.num_inference_steps)
        dt = 1.0 / max(num_steps, 1)
        camera = hasattr(self.model, 'camera_grounded_query')
        previous_metric = torch.zeros_like(current_actions) if camera else None
        if return_trace:
            trace_states, trace_clean, trace_previous, trace_velocity = ([current_actions], [], [], [])
        for step in range(num_steps):
            if query_prefixes is not None:
                previous_metric = query_prefixes[:, step]
            t_cont = torch.full((B,), step * dt, device=device, dtype=current_actions.dtype)
            pred_flow = self.predict_flow_velocity(current_actions, t_cont, vl_embeds, history_embeds, ego_embeds, encoder_attention_mask, action_input.get('vehicle_geometry'), **{'camera_inputs': action_input, 'previous_metric': previous_metric} if camera else {})
            if camera or return_trace:
                clean_estimate = current_actions + (1 - step * dt) * pred_flow
            if return_trace:
                trace_clean.append(clean_estimate)
                trace_previous.append(previous_metric if camera else torch.zeros_like(current_actions))
                trace_velocity.append(pred_flow)
            if camera and (not fixed_first_estimate or step == 0):
                previous_metric = self.denorm_odo(clean_estimate.clamp(-1, 1)).detach()
            current_actions = current_actions + dt * pred_flow
            if return_trace:
                trace_states.append(current_actions)
        final_action_clip_value = getattr(self, 'final_action_clip_value', 1.0)
        if final_action_clip_value is not None:
            if return_trace:
                current_actions = current_actions.clamp(-final_action_clip_value, final_action_clip_value)
            else:
                current_actions.clamp_(-final_action_clip_value, final_action_clip_value)
        final_actions = self.denorm_odo(current_actions)
        out: Dict[str, Any] = {'pred_traj': final_actions}
        if return_trace:
            out.update(flow_trace_states=torch.stack(trace_states, dim=1), flow_trace_clean=torch.stack(trace_clean, dim=1), flow_trace_previous=torch.stack(trace_previous, dim=1).detach(), flow_trace_velocity=torch.stack(trace_velocity, dim=1))
        if bool(getattr(self.config, 'record_delta_pi_curve', False)) and hasattr(self, '_delta_pi_curve'):
            if len(self._delta_pi_curve) > 0:
                curve = torch.stack(self._delta_pi_curve, dim=0)
                out['delta_pi_curve'] = curve
                out['delta_pi_mean'] = curve.mean()
        return BatchFeature(data=out)

    def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalizes trajectory coordinates and heading to the range [-1, 1]."""
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)

    def denorm_odo(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
