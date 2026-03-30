# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
PhysBrainVLA implementation

Architecture combining TwinBrainVLA and LangForce:
- TwinBrainVLA: dual-brain architecture (frozen left brain + trainable right brain), connected via MoT
- LangForce: Bayesian decomposition with LLR regularization to prevent visual shortcuts

Core design:
- Left brain always processes (V + L), without action tokens
- Right brain handles two types of input:
  - Prior: (V + A + L) - action token placed before language
  - Posterior: (V + L + A) - action token placed after language
- LLR loss: log p(L|V,A_prior) - sg(log p(L|V)), preventing the right brain from over-relying on left brain KV and ignoring language
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional, Tuple, Set
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, Qwen2_5_VLCausalLMOutputWithPast
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers import AutoProcessor

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

try:
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    # Try to import Qwen3 specific classes if available in current environment
    # Note: As of early versions, these might need specific import paths
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False
    logger.warning("Qwen3VL not found, MoT interface will only support Qwen2.5-VL")

try:
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb as apply_rotary_pos_emb_qwen3_5,
        eager_attention_forward as eager_attention_forward_qwen3_5,
    )
    QWEN3_5_AVAILABLE = True
except ImportError:
    QWEN3_5_AVAILABLE = False
    Qwen3_5ForConditionalGeneration = None
    logger.warning("Qwen3.5 not found, MoT interface will not support Qwen3.5")

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# ===== Qwen special tokens =====
VISION_START_TOKEN_INDEX = 151652  # <|vision_start|>
VISION_END_TOKEN_INDEX   = 151654  # <|vision_end|>
IMAGE_TOKEN_INDEX        = 151655  # <|image_pad|>
VIDEO_TOKEN_INDEX        = 151656  # <|video_pad|>
IM_START_TOKEN_INDEX     = 151644  # <|im_start|>
IM_END_TOKEN_INDEX       = 151645  # <|im_end|>

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

# related configuration and utils
from starVLA.model.framework.physbrainvla_conf.configuration_physbrain_vla import DoubleVLAConfig, DEFUALT_RIGHT_VLM_USER_PROMPT_TEMPLATE, RIGHT_VLM_USER_PROMPT_TEMPLATE_NO_STATE, DEFUALT_LEFT_VLM_SYSTEM_PROMPT, DEFAULT_RIGHT_VLM_SYSTEM_PROMPT, FIXED_STATE_DIM
from starVLA.model.framework.physbrainvla_conf import utils


# ----------------------------
# MoT / Double VLM Logic
# ----------------------------

class MoT_Qwen_VL_Interface(nn.Module):
    """
    Mixture-of-Transformers Interface for Qwen2.5-VL.
    Maintains a frozen 'anchor' VLM and a trainable 'expert' VLM.
    They interact via joint attention in the Transformer layers.
    """
    def __init__(self, config: OmegaConf, vla_config: DoubleVLAConfig):
        super().__init__()
        
        # config
        self.config = config
        self.vla_config = vla_config
        
        # Load two independent VLM interfaces
        # Note: This will load weights twice. Ensure GPU memory is sufficient.
        self.frozen_vlm = get_vlm_model(config)
        self.trainable_vlm = get_vlm_model(config)
        
        if not self.vla_config.freeze_left_vlm:
            print("🟢 Left VLM is not frozen.")

        # Freeze the first VLM
        if self.vla_config.freeze_left_vlm:
            utils.freeze_qwen_vl(self.frozen_vlm)
        
        # Trainable VLM remains trainable
        self.trainable_vlm.model.train()
        
        # Expose attributes expected by the framework
        self.model = self.trainable_vlm.model
        self.processor = self.trainable_vlm.processor
        
        # Detect Model Type
        self.is_qwen3 = QWEN3_AVAILABLE and isinstance(self.model, Qwen3VLForConditionalGeneration)
        if self.is_qwen3:
            logger.info("MoT Interface: Detected Qwen3-VL model.")
        self.is_qwen3_5 = QWEN3_5_AVAILABLE and isinstance(self.model, Qwen3_5ForConditionalGeneration)
        if self.is_qwen3_5:
            logger.info("MoT Interface: Detected Qwen3.5 model (hybrid linear+full attention with gated output).")  
        
        # Add special tokens to RightVLM
        self._add_special_tokens_to_right_vlm()

        # Determine Attention Mode
        # 'full_joint': Bidirectional awareness (Frozen <-> Trainable)
        # 'unidirectional': Frozen only sees Frozen. Trainable sees (Frozen + Trainable)
        self.mot_attention_mode = getattr(config, 'doublevla.mot_attention_mode', 'unidirectional')
        logger.info(f"🟢 MoT Attention Mode: {self.mot_attention_mode}")

        # Cache for causal masks keyed by (device, dtype, seq_len)
        # Helps reduce per-step overhead when seq_len is stable.
        self._causal_mask_cache: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
        
        self.action_query_token_num = getattr(self.vla_config, 'action_query_token_num', 8)
        
        # Cache for <|im_end|> token id (used in LLR computation)
        self._im_end_id = None
    
    def _add_special_tokens_to_right_vlm(self):
        tokenizer = self.trainable_vlm.processor.tokenizer
        model = self.trainable_vlm.model

        # If state is not used (state_style="none") and <|action|> is not actually used
        # in the current implementation, skip adding extra special tokens to tokenizer/embeddings.
        # if self.vla_config.state_style == "none":
        #     logger.info("State style is 'none' and action_query_token is unused in current implementation; skip adding special tokens to RightVLM.")
        #     self.state_token_id = None
        #     self.action_token_id = None
        #     return

        # Define special tokens to be added
        special_tokens_dict = {
            'additional_special_tokens': [
                DoubleVLAConfig.state_special_token,   # <|propri|>
                DoubleVLAConfig.action_query_token,    # <|action|>
            ]
        }
    
        # Check if tokens already exist (avoid duplicate additions)
        existing_special_tokens = getattr(tokenizer, 'additional_special_tokens', None) or []
        new_tokens = [
            t for t in special_tokens_dict['additional_special_tokens'] 
            if t not in existing_special_tokens
        ]
        
        if not new_tokens:
            logger.info("Special tokens already exist in tokenizer, skipping...")
        else:
            # Add new tokens to tokenizer
            # Record original size
            old_vocab_size = model.get_input_embeddings().weight.shape[0]
            
            num_added = tokenizer.add_special_tokens({
                'additional_special_tokens': existing_special_tokens + new_tokens
            })
            logger.info(f"🟢 Added {num_added} special tokens to trainable VLM tokenizer: {new_tokens}")
            
            # Compute new size from the actual number of added tokens
            # instead of relying on len(tokenizer), which can be inaccurate.
            new_size = old_vocab_size + num_added
            model.resize_token_embeddings(new_size)
            new_vocab_size = model.get_input_embeddings().weight.shape[0]
            logger.info(f"🟢 Resized embedding: {old_vocab_size} -> {new_vocab_size} (added {num_added} tokens)")
            # Optional: initialize new token embeddings with mean values
            # (more stable than random initialization).
            self._init_new_token_embeddings(model, old_vocab_size, new_vocab_size)
        
        # Save token IDs for later use
        self.state_token_id = tokenizer.convert_tokens_to_ids(DoubleVLAConfig.state_special_token)
        self.action_token_id = tokenizer.convert_tokens_to_ids(DoubleVLAConfig.action_query_token)
        
        logger.info(f"🟢 State token ID: {self.state_token_id}, Action token ID: {self.action_token_id}")
    
    def _init_new_token_embeddings(self, model, old_vocab_size: int, new_vocab_size: int):
        """
        Initialize new token embeddings with the mean of existing embeddings.
        This is typically more stable than random initialization and converges faster.
        """
        if old_vocab_size >= new_vocab_size:
            return  # No new tokens added
        with torch.no_grad():
            # Get input embeddings
            input_embeddings = model.get_input_embeddings()
        
            # Compute mean of existing embeddings
            old_embeddings = input_embeddings.weight[:old_vocab_size]
            mean_embedding = old_embeddings.mean(dim=0)
        
            # Initialize new tokens with mean embedding
            for i in range(old_vocab_size, new_vocab_size):
                input_embeddings.weight[i] = mean_embedding.clone()
        
            # If model has output embeddings (usually shared when weights are tied)
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None and output_embeddings.weight.shape[0] == new_vocab_size:
                old_out_embeddings = output_embeddings.weight[:old_vocab_size]
                mean_out_embedding = old_out_embeddings.mean(dim=0)
                for i in range(old_vocab_size, new_vocab_size):
                    output_embeddings.weight[i] = mean_out_embedding.clone()
        
            logger.info(f"🟢 Initialized new token embeddings with mean of existing embeddings")
            

    def build_qwenvl_inputs(self, images, instructions, state=None, embodiment_id=None, mode='posterior'):
        """
        Build inputs for both VLMs.
        Frozen VLM: images + instructions only (always V + L, no action token)
        Trainable VLM: depends on mode
          - 'prior': (V + A + L) - action token BEFORE language
          - 'posterior': (V + L + A) - action token AFTER language
        
        Args:
            images: List of image lists [B, [PIL Images]]
            instructions: List of instruction strings [B, str]
            state: Optional tensor of shape [B, 1, state_dim] or None
            embodiment_id: Optional list of embodiment IDs [B] or None
            mode: 'prior' or 'posterior', determines action token position
        
        Returns:
            frozen_inputs: dict for frozen VLM
            trainable_inputs: dict for trainable VLM
            state_tensor: processed state tensor or None
            embodiment_id_tensor: processed embodiment ID tensor or None
        """
        # Frozen VLM: images + instructions only (always V + L, no action token)
        frozen_vlm_system_prompt = (
            "You are a helpful robot brain that can understand images and texts.\n"
            "You will be provided with multiple observation images and an instruction. You should understand these images and instructions."
        ).strip()
        frozen_inputs = self.frozen_vlm.build_qwenvl_inputs(images, instructions, system_prompt=frozen_vlm_system_prompt)
        
        # # For LIBERO evaluation, enable this line if needed.
        # embodiment_id = [25 for i in range(len(instructions))]
        
        robot_types = [utils.get_robot_type(eid) for eid in embodiment_id] if embodiment_id is not None else None
        
        
        # Trainable VLM: depends on mode
        action_query_token_num = getattr(self.vla_config, 'action_query_token_num', 1)
        action_query_tokens_str = " ".join([DoubleVLAConfig.action_query_token] * action_query_token_num)
        
        if mode == 'prior':
            # Prior mode: (V + A + L) - action token BEFORE language
            trainable_instruction_template = (
                "Action Query: " + action_query_tokens_str + "\n"
                "Instruction: {inst}"
            ).strip()
        else:
            # Posterior mode: (V + L + A) - action token AFTER language
            trainable_instruction_template = (
                "Instruction: {inst}\n"
                "Action Query: " + action_query_tokens_str
            ).strip()
        
        trainable_instructions = [
            trainable_instruction_template.format(inst=inst) for inst in instructions
        ]

        trainable_vlm_system_prompt = (
            "You are a helpful robot. You need to encode the special token <|action|> into hidden states that serve to control the Action Expert."
        ).strip()
        
        trainable_inputs = self.trainable_vlm.build_qwenvl_inputs(images, trainable_instructions, system_prompt=trainable_vlm_system_prompt)
        
        
        return frozen_inputs, trainable_inputs, None, None  # state_tensor and embodiment_id_tensor are set to None; this implementation does not use them.
    
    

    def forward(self, is_vla: bool = False, frozen_inputs=None, trainable_inputs=None, state_tensor=None, embodiment_id_tensor=None, selected_layer: int = -1, **kwargs):
        """
        Joint forward pass with separate inputs for frozen and trainable VLMs.
        
        Args:
            frozen_inputs: dict of inputs for frozen VLM (vision + text only)
            trainable_inputs: dict of inputs for trainable VLM (vision + text, state added later)
            state_tensor: Optional [B, 1, state_dim] state embeddings
            embodiment_id_tensor: Optional [B] embodiment ID tensor
            selected_layer: int, specifies which layer output to return.
                           - Positive: return output of layer `selected_layer` (0-indexed)
                           - Negative: -1 means last layer, -2 means second-to-last, and so on
                           - Default is -1 (last layer)
            **kwargs: Legacy support for single-input mode
        """
        # Legacy support: if no separate inputs provided, use kwargs (old behavior)
        if frozen_inputs is None and trainable_inputs is None:
            frozen_inputs = kwargs
            trainable_inputs = kwargs
            state_tensor = None
        
        # ========== VLM co-training mode ==========
        # Use trainable VLM's standard forward directly.
        # Benefits:
        # 1. Avoids image token/feature mismatch caused by input_ids truncation
        #    (VLM collator truncates input_ids to model_max_length but keeps all
        #     pixel_values/image_grid_thw, causing get_placeholder_mask to fail)
        # 2. Skips unnecessary frozen VLM processing for VLM tasks
        # 3. Avoids action token extraction warnings (VLM data has no <|action|> tokens)
        if not is_vla:
            vlm_inputs = self._fix_truncated_vlm_inputs(trainable_inputs)
            
            # Prepare clean inputs for Qwen model (only keep valid keys)
            valid_keys = {
                'input_ids', 'attention_mask', 'position_ids', 'labels',
                'pixel_values', 'pixel_values_videos', 
                'image_grid_thw', 'video_grid_thw',
                'inputs_embeds', 'past_key_values', 'use_cache',
                'output_attentions', 'output_hidden_states', 'return_dict'
            }
            clean_inputs = {k: v for k, v in vlm_inputs.items() if k in valid_keys and v is not None}
            
            # Ensure labels are properly shaped and don't contain invalid values
            if 'labels' in clean_inputs:
                labels = clean_inputs['labels']
                # Get vocab_size safely (Qwen3 has different config structure)
                try:
                    vocab_size = self.trainable_vlm.model.config.vocab_size
                except AttributeError:
                    # Qwen3VL case: vocab_size in language_config
                    try:
                        vocab_size = self.trainable_vlm.model.config.language_config.vocab_size
                    except AttributeError:
                        # Fallback: get from embedding layer
                        vocab_size = self.trainable_vlm.model.get_input_embeddings().weight.shape[0]
                        logger.warning(f"Could not find vocab_size in config, using embedding size: {vocab_size}")
                
                # Check for out-of-range labels (keep IGNORE_INDEX=-100 which is valid)
                invalid_mask = (labels >= vocab_size) & (labels != IGNORE_INDEX)
                if invalid_mask.any():
                    logger.error(f"Found {invalid_mask.sum().item()} labels >= {vocab_size}, clamping to IGNORE_INDEX")
                    labels = labels.clone()
                    labels[invalid_mask] = IGNORE_INDEX
                    clean_inputs['labels'] = labels
            
            return self.trainable_vlm.model(**clean_inputs)
        
        # ========== VLA mode (MoT joint attention) ==========
        # Whether to freeze LeftVLM (for MoT ablation)
        freeze_left = self.vla_config.freeze_left_vlm
        # print("🟢 Left Brain Freezed: ", freeze_left)
        
        # Get action query token number
        action_query_token_num = getattr(self.vla_config, 'action_query_token_num', 1)

        # Extract common metadata (they should be identical for both VLMs)
        # pixel_values = frozen_inputs.get("pixel_values")
        # pixel_values_videos = frozen_inputs.get("pixel_values_videos")
        # image_grid_thw = frozen_inputs.get("image_grid_thw")
        # video_grid_thw = frozen_inputs.get("video_grid_thw")
        
        # --- Run Preprocessing for Frozen Model (Vision + Text only) ---
        if freeze_left:
            # No gradients needed when LeftVLM is frozen
            with torch.no_grad():
                frozen_outputs_pre = self._preprocess_inputs(
                    self.frozen_vlm.model.model, 
                    **frozen_inputs
                )
        else:
            # Allow gradients when LeftVLM is unfrozen for ablation
            frozen_outputs_pre = self._preprocess_inputs(
                self.frozen_vlm.model.model, 
                **frozen_inputs
            )
        (frozen_embeds, frozen_pos_ids, frozen_rotary_emb, frozen_mask, frozen_kv_cache) = frozen_outputs_pre
        
        # --- Run Preprocessing for Trainable Model (Vision + Text + State) ---
        trainable_outputs_pre = self._preprocess_inputs(
            self.trainable_vlm.model.model, 
            **trainable_inputs
        )
        (trainable_embeds, trainable_pos_ids, trainable_rotary_emb, trainable_mask, trainable_kv_cache) = trainable_outputs_pre

        # --- Joint Transformer Loop ---
        # Iterate layers
        frozen_hidden = frozen_embeds
        trainable_hidden = trainable_embeds
        
        # Access layers generically
        if self.is_qwen3: 
             # Qwen3VL structure: model.model.language_model.layers
             frozen_layers = self.frozen_vlm.model.model.language_model.layers
             trainable_layers = self.trainable_vlm.model.model.language_model.layers
        else:
             frozen_layers = self.frozen_vlm.model.model.language_model.layers
             trainable_layers = self.trainable_vlm.model.model.language_model.layers
        
        # Compute effective target layer index (supports negative indexing)
        num_layers = len(trainable_layers)
        target_layer_idx = selected_layer if selected_layer >= 0 else num_layers + selected_layer
        target_layer_idx = max(0, min(target_layer_idx, num_layers - 1))  # clamp to valid range
        selected_hidden_state = None  # Stores hidden state from the selected layer
        
        # Determine attention implementation
        # We force 'eager' or utilize what's available, but we need to control the attention func
        # to handle joint KV.
        # If we use flash_attn, we need to construct proper packed KV.
        
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward

        for i, (l_frozen, l_trainable) in enumerate(zip(frozen_layers, trainable_layers)):
            # ============ Frozen Stream ============
            if freeze_left:
                # Frozen LeftVLM: compute fully under no_grad and detach from graph
                with torch.no_grad():
                    frozen_residual = frozen_hidden
                    frozen_hidden_normed = l_frozen.input_layernorm(frozen_hidden)
                    f_q, f_k, f_v, f_gate = self._project_qkv(l_frozen.self_attn, frozen_hidden_normed)

                # Detach to ensure no gradient connection even if used in joint attention
                f_k = f_k.detach()
                f_v = f_v.detach()
                if f_gate is not None:
                    f_gate = f_gate.detach()
            else:
                # Unfrozen LeftVLM: normal compute path with gradient flow
                frozen_residual = frozen_hidden
                frozen_hidden_normed = l_frozen.input_layernorm(frozen_hidden)
                f_q, f_k, f_v, f_gate = self._project_qkv(l_frozen.self_attn, frozen_hidden_normed)
            
            # ============ Trainable Stream ============
            trainable_residual = trainable_hidden
            trainable_hidden_normed = l_trainable.input_layernorm(trainable_hidden)
            t_q, t_k, t_v, t_gate = self._project_qkv(l_trainable.self_attn, trainable_hidden_normed)
            
            # Apply RoPE
            # Note: Qwen2.5-VL applies RoPE inside forward usually. We do it manually.
            # Assuming shared or identical pos_ids
            # frozen_pos_ids should be used for frozen states, etc.
            
            # Helper to apply RoPE with specified rotary embeddings
            def apply_rope(q, k, v, rotary_emb, rope_scaling=None):
                """
                Apply Rotary Position Embedding (RoPE) to query and key tensors.
                
                Args:
                    q, k, v: Query, Key, Value tensors [B, n_heads, L, head_dim]
                    rotary_emb: Tuple of (cos, sin) for RoPE
                    rope_scaling: Rope scaling config (for Qwen2.5-VL multimodal RoPE)
                
                Returns:
                    q, k, v: Tensors with RoPE applied to q and k
                """
                cos, sin = rotary_emb
                
                if self.is_qwen3_5:
                    # Qwen3.5 uses its own apply_rotary_pos_emb with partial rotation factor
                    q, k = apply_rotary_pos_emb_qwen3_5(q, k, cos, sin)
                elif self.is_qwen3:
                    # Qwen3 uses generic apply_rotary_pos_emb(q, k, cos, sin)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                else:
                    # Qwen2.5 uses multimodal variant
                    q, k = apply_multimodal_rotary_pos_emb(
                        q, k, cos, sin, rope_scaling["mrope_section"]
                    )
                return q, k, v
            
            # Note: We assume rotary embeddings are compatible (same config).
            # Only Qwen2.5-VL uses rope_scaling with mrope_section
            # Qwen3-VL and Qwen3.5 do not have rope_scaling attribute
            def get_rope_scaling(attn_module):
                """Safely get rope_scaling config, return None if not available."""
                return getattr(attn_module, 'rope_scaling', None)
            
            frozen_rope_scaling = get_rope_scaling(l_frozen.self_attn)
            trainable_rope_scaling = get_rope_scaling(l_trainable.self_attn)
            
            # Apply RoPE to frozen stream using frozen_rotary_emb
            if freeze_left:
                # Frozen mode: no gradients needed
                with torch.no_grad():
                    f_q, f_k, f_v = apply_rope(
                        f_q, f_k, f_v, 
                        frozen_rotary_emb,
                        rope_scaling=frozen_rope_scaling
                    )
                # Re-detach after RoPE to ensure clean separation
                f_q = f_q.detach()
                f_k = f_k.detach()
                # f_v doesn't change from RoPE, but ensure it's detached
                f_v = f_v.detach()
            else:
                # Unfrozen mode: keep gradients
                f_q, f_k, f_v = apply_rope(
                    f_q, f_k, f_v,
                    frozen_rotary_emb,
                    rope_scaling=frozen_rope_scaling
                )
            
            # Apply RoPE to trainable stream using trainable_rotary_emb
            t_q, t_k, t_v = apply_rope(
                t_q, t_k, t_v,
                trainable_rotary_emb,
                rope_scaling=trainable_rope_scaling
            )

            # Check mode (defaults to self.mot_attention_mode, which defaults to 'unidirectional' if not in config)
            # Allow override via kwargs for experiments
            current_mode = kwargs.get('mot_attention_mode', self.mot_attention_mode)
            
            if current_mode == 'unidirectional':
                # Mode: Frozen model is independent (Self-Attention only).
                #       Trainable model attends to Frozen + Trainable.
                
                # 1. Frozen Stream Implementation
                f_attn_out, _ = eager_attention_forward(
                    l_frozen.self_attn, f_q, f_k, f_v, frozen_mask, 
                    scaling=l_frozen.self_attn.scaling,
                    dropout=0.0
                )

                # 2. Trainable Stream Implementation (Joint)
                # Note: trainable and frozen sequences may have different lengths
                # because trainable has <STATE> token
                joint_k = torch.cat([f_k, t_k], dim=2)
                joint_v = torch.cat([f_v, t_v], dim=2)
                
                if frozen_mask is not None and trainable_mask is not None:
                    # Handle potential length mismatch between frozen and trainable masks
                    frozen_len = frozen_mask.shape[-1]
                    trainable_len = trainable_mask.shape[-1]
                    
                    if frozen_len != trainable_len:
                        # Pad frozen_mask to match trainable_mask query dimension
                        # frozen_mask: [B, 1, L_frozen, L_frozen]
                        # trainable_mask: [B, 1, L_trainable, L_trainable]
                        # We need joint_mask: [B, 1, L_trainable, L_frozen + L_trainable]
                        
                        # Extend frozen_mask to be [B, 1, L_trainable, L_frozen]
                        # by padding with -inf (can't attend to non-existent positions)
                        pad_size = trainable_len - frozen_len
                        frozen_mask_extended = torch.full(
                            (frozen_mask.shape[0], frozen_mask.shape[1], trainable_len, frozen_len),
                            float('-inf'),
                            device=frozen_mask.device,
                            dtype=frozen_mask.dtype
                        )
                        # Copy the original frozen_mask to top-left corner
                        frozen_mask_extended[:, :, :frozen_len, :] = frozen_mask
                        # Extra rows (for extra tokens in trainable) can see all frozen tokens
                        frozen_mask_extended[:, :, frozen_len:, :] = 0.0
                        
                        joint_mask = torch.cat([frozen_mask_extended, trainable_mask], dim=-1)
                    else:
                        # Same length, simple concatenation
                        joint_mask = torch.cat([frozen_mask, trainable_mask], dim=-1)
                else:
                    joint_mask = None
                    
                t_attn_out, _ = eager_attention_forward(
                    l_trainable.self_attn, t_q, joint_k, joint_v, joint_mask,
                    scaling=l_trainable.self_attn.scaling,
                    dropout=l_trainable.self_attn.attention_dropout if self.training else 0.0
                )
            
            else:
                # Mode: 'full_joint' (Original behavior)
                #       Both streams see everything.
                
                # Joint mechanism: Concatenate Keys and Values
                joint_k = torch.cat([f_k, t_k], dim=2) 
                joint_v = torch.cat([f_v, t_v], dim=2)
                
                # Prepare masks for both streams (handle length mismatch)
                if frozen_mask is not None and trainable_mask is not None:
                    frozen_len = frozen_mask.shape[-1]
                    trainable_len = trainable_mask.shape[-1]
                    
                    if frozen_len != trainable_len:
                        # For frozen: extend to match its KV length, then concat with trainable KV mask
                        # frozen_mask: [B, 1, L_frozen, L_frozen] -> [B, 1, L_frozen, L_frozen + L_trainable]
                        frozen_joint_mask = torch.cat([
                            frozen_mask,
                            torch.zeros(
                                frozen_mask.shape[0], frozen_mask.shape[1], frozen_len, trainable_len,
                                device=frozen_mask.device, dtype=frozen_mask.dtype
                            )
                        ], dim=-1)
                        
                        # For trainable: extend to match its query length, then concat
                        # trainable_mask: [B, 1, L_trainable, L_trainable] -> [B, 1, L_trainable, L_frozen + L_trainable]
                        frozen_mask_extended = torch.full(
                            (trainable_mask.shape[0], trainable_mask.shape[1], trainable_len, frozen_len),
                            0.0,  # Trainable can see all frozen tokens
                            device=trainable_mask.device,
                            dtype=trainable_mask.dtype
                        )
                        # Extra rows in trainable can see all frozen tokens
                        trainable_joint_mask = torch.cat([frozen_mask_extended, trainable_mask], dim=-1)
                    else:
                        # Same length, simple concatenation for both
                        frozen_joint_mask = torch.cat([frozen_mask, trainable_mask], dim=-1)
                        trainable_joint_mask = frozen_joint_mask
                else:
                    frozen_joint_mask = None
                    trainable_joint_mask = None

                # Frozen Attn
                f_attn_out, _ = eager_attention_forward(
                    l_frozen.self_attn, f_q, joint_k, joint_v, frozen_joint_mask, 
                    scaling=l_frozen.self_attn.scaling,
                    dropout=0.0 # No dropout for frozen inference
                )
                
                # Trainable Attn
                t_attn_out, _ = eager_attention_forward(
                    l_trainable.self_attn, t_q, joint_k, joint_v, trainable_joint_mask,
                    scaling=l_trainable.self_attn.scaling,
                    dropout=l_trainable.self_attn.attention_dropout if self.training else 0.0
                )
            
            # ============ Output Projections ============
            # Frozen stream
            if freeze_left:
                with torch.no_grad():
                    f_attn_out = f_attn_out.reshape(f_attn_out.shape[0], f_attn_out.shape[1], -1).contiguous()
                    if f_gate is not None:  # Qwen3.5: gated attention output (gate * sigmoid)
                        f_attn_out = f_attn_out * torch.sigmoid(f_gate)
                    f_attn_out = l_frozen.self_attn.o_proj(f_attn_out)
                    frozen_hidden = frozen_residual + f_attn_out
            else:
                f_attn_out = f_attn_out.reshape(f_attn_out.shape[0], f_attn_out.shape[1], -1).contiguous()
                if f_gate is not None:  # Qwen3.5: gated attention output
                    f_attn_out = f_attn_out * torch.sigmoid(f_gate)
                f_attn_out = l_frozen.self_attn.o_proj(f_attn_out)
                frozen_hidden = frozen_residual + f_attn_out
            
            # Trainable stream
            t_attn_out = t_attn_out.reshape(t_attn_out.shape[0], t_attn_out.shape[1], -1).contiguous()
            if t_gate is not None:  # Qwen3.5: gated attention output
                t_attn_out = t_attn_out * torch.sigmoid(t_gate)
            t_attn_out = l_trainable.self_attn.o_proj(t_attn_out)
            trainable_hidden = trainable_residual + t_attn_out
            
            # ============ Feed Forward (Independent) ============
            # Frozen FFN
            if freeze_left:
                with torch.no_grad():
                    frozen_residual = frozen_hidden
                    frozen_hidden = l_frozen.post_attention_layernorm(frozen_hidden)
                    frozen_hidden = l_frozen.mlp(frozen_hidden)
                    frozen_hidden = frozen_residual + frozen_hidden
            else:
                frozen_residual = frozen_hidden
                frozen_hidden = l_frozen.post_attention_layernorm(frozen_hidden)
                frozen_hidden = l_frozen.mlp(frozen_hidden)
                frozen_hidden = frozen_residual + frozen_hidden
            
            # Trainable FFN
            trainable_residual = trainable_hidden
            trainable_hidden = l_trainable.post_attention_layernorm(trainable_hidden)
            trainable_hidden = l_trainable.mlp(trainable_hidden)
            trainable_hidden = trainable_residual + trainable_hidden
            
            # Save selected-layer hidden state (after FFN, before next layer)
            if i == target_layer_idx:
                # No need to clone: trainable_hidden is re-bound each layer (no in-place ops)
                selected_hidden_state = trainable_hidden
            
        # Final Norms
        frozen_hidden = self.frozen_vlm.model.model.language_model.norm(frozen_hidden)
        trainable_hidden = self.trainable_vlm.model.model.language_model.norm(trainable_hidden)
        
        # Apply final norm to selected_hidden_state as well (if not last layer)
        if selected_hidden_state is not None and target_layer_idx < num_layers - 1:
            # Intermediate layer hidden states also need normalization for downstream use
            selected_hidden_state = self.trainable_vlm.model.model.language_model.norm(selected_hidden_state)
        else:
            # For last layer, use trainable_hidden directly (already normalized)
            selected_hidden_state = trainable_hidden
        
        # Extract action query token hidden states
        # Find positions of action query tokens in the input sequence
        input_ids = trainable_inputs['input_ids']  # [B, L]
        action_pos_mask = (input_ids == self.action_token_id)  # [B, L]

        # Extract hidden states at action query token positions without assuming they are at the very end (handles Right Padding)
        B, L, H = selected_hidden_state.shape
        
        # Check if we have the expected number of action tokens per sample
        # This is critical because if padding or truncation happened unexpectedly, count might differ
        per_sample_counts = action_pos_mask.sum(dim=1) # [B]
        
        if (per_sample_counts == action_query_token_num).all():
            # Robust extraction: Select exactly the tokens corresponding to action_token_id
            # selected_hidden_state[action_pos_mask] returns flattened tensor of shape [B * N, H]
            # We then reshape it back to [B, N, H]
            action_query_hidden_states = selected_hidden_state[action_pos_mask].view(B, action_query_token_num, H)
        else:
            # Determine padding side to make a best guess if counts don't match (e.g. truncation)
            # Or if some samples have different counts (unexpected bug)
            logger.warning(
                f"Warning: Action token count mismatch! "
                f"Expected {action_query_token_num}, but got {per_sample_counts.tolist()}. "
                f"Falling back to taking the last {action_query_token_num} tokens, which might be incorrect if Right Padding is used."
            )
            action_query_hidden_states = selected_hidden_state[:, -action_query_token_num:, :]

        # Verify output shape
        # action_query_hidden_states: [B, N, H]
        
        # Return outputs. We package it like Qwen output so existing code works.
        # Framework expects `qwenvl_outputs.hidden_states[-1]}` or `qwenvl_outputs.selected_hidden_state`
        
        # We return only the action query token states as the primary output for Action Head
        # This replaces the previous behavior of returning all token states
        
        # output_cls = Qwen2_5_VLModelOutputWithPast
        # if self.is_qwen3 and QWEN3_AVAILABLE:
        #     output_cls = Qwen3VLModelOutputWithPast

        #########
        # return
        # Three output modes: VLA outputs, Qwen2.5 outputs, and Qwen3 outputs
        #########
        if is_vla:
            # Compute logits for LLR computation (from trainable VLM)
            logits = self.trainable_vlm.model.lm_head(trainable_hidden)  # [B, L, V]
            
            return BaseModelOutputWithPast(
                last_hidden_state=action_query_hidden_states,  # [B, N, H] - Only action query token hidden states
                hidden_states=[action_query_hidden_states, trainable_hidden],  # [action_hidden, full_hidden]
                past_key_values=None,
            ), logits  # Return logits for LLR computation
        else:
            # VLM mode: return standard Qwen outputs for VQA training
            # Use trainable_hidden as the main output (for language modeling loss)
            logits = self.trainable_vlm.model.lm_head(trainable_hidden)
            
            # Get labels from trainable_inputs if available
            labels = trainable_inputs.get('labels', None)
            
            # Compute language modeling loss if labels provided
            loss = None
            if labels is not None:
                # Shift logits and labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            
            # Return in Qwen output format
            if self.is_qwen3 and QWEN3_AVAILABLE:
                return Qwen3VLCausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=None,
                    hidden_states=(trainable_hidden,) if True else None,
                    attentions=None,
                )
            else:
                return Qwen2_5_VLCausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=None,
                    hidden_states=(trainable_hidden,) if True else None,
                    attentions=None,
                )


    def _preprocess_inputs(self, model_module, **kwargs):
        """Helper to run embeddings and RoPE prep"""
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        
        # Initialize image_mask to None (will be set if pixel_values exist)
        image_mask = None
        
        # Embeddings logic from Qwen2_5_VLModel.forward
        if inputs_embeds is None:
            inputs_embeds = model_module.get_input_embeddings()(input_ids)
            
        if pixel_values is not None:
            image_embeds = model_module.get_image_features(
                pixel_values=pixel_values, 
                image_grid_thw=image_grid_thw
            )

            # Handle model output dataclass objects (e.g., BaseModelOutputWithDeepstackFeatures
            # returned by newer Qwen3/Qwen3.5-VL). Extract the primary feature tensor.
            if not isinstance(image_embeds, (torch.Tensor, list, tuple)):
                if hasattr(image_embeds, 'last_hidden_state'):
                    image_embeds = image_embeds.last_hidden_state
                elif hasattr(image_embeds, 'image_features'):
                    image_embeds = image_embeds.image_features
                else:
                    # Fallback: try the first value in the object's dict
                    vals = [v for v in vars(image_embeds).values() if isinstance(v, torch.Tensor)]
                    if vals:
                        image_embeds = vals[0]
                    else:
                        raise TypeError(
                            f"get_image_features returned an unexpected type {type(image_embeds)} "
                            "with no recognizable tensor attribute."
                        )

            # Qwen3-VL returns (image_embeds, deepstack_image_embeds), only need the first part
            if isinstance(image_embeds, tuple) and len(image_embeds) == 2 and not isinstance(image_embeds[0], torch.Tensor):
                image_embeds = image_embeds[0]
            elif self.is_qwen3 and isinstance(image_embeds, tuple) and len(image_embeds) == 2:
                image_embeds = image_embeds[0]

            # Qwen3-VL compatibility: get_image_features might return list of tuples (feature, grid)
            if isinstance(image_embeds, (list, tuple)) and len(image_embeds) > 0 and isinstance(image_embeds[0], (list, tuple)):
                image_embeds = [x[0] for x in image_embeds]

            # cat only when we have a list/tuple of tensors; single tensors are used directly
            if isinstance(image_embeds, torch.Tensor):
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            else:
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            # Qwen3-VL spatial merger fix:
            # get_image_features may return raw ViT patch features (N_raw) extracted from
            # last_hidden_state, while input_ids has N_merged = N_raw / merge_size^2 image_pad
            # tokens.  We must apply the visual merger to align counts before calling
            # get_placeholder_mask, which enforces features == tokens.
            image_token_id_local = getattr(
                self.trainable_vlm.model.config, 'image_token_id', IMAGE_TOKEN_INDEX
            )
            n_expected = int((input_ids == image_token_id_local).sum().item())
            if image_embeds.shape[0] != n_expected and image_embeds.shape[0] > n_expected:
                # Try the visual merger in the model_module (Qwen3-VL: model_module.visual.merger)
                if hasattr(model_module, 'visual') and hasattr(model_module.visual, 'merger'):
                    image_embeds = model_module.visual.merger(image_embeds)
                else:
                    logger.warning(
                        f"image_embeds count mismatch: got {image_embeds.shape[0]}, "
                        f"expected {n_expected}. No visual.merger found; "
                        "get_placeholder_mask may fail."
                    )

            image_mask, _ = model_module.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Position IDs
        # We assume standard call
        # Note: We need to handle `rope_deltas` state if we were doing caching. Simplified here.
        if hasattr(model_module, "get_rope_index"):
            # Qwen2.5 and Qwen3 both have this
            position_ids, rope_deltas = model_module.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask
            )
        else:
            # Fallback? Unlikely to hit this if using Qwen
            position_ids = None 
            rope_deltas = None
        
        # Create Rotation Embeddings
        # rot_emb outputs cos, sin
        if hasattr(model_module.language_model, "rotary_emb"):
            rotary_emb = model_module.language_model.rotary_emb(inputs_embeds, position_ids)
        elif hasattr(model_module, "rotary_emb"):
             # Fallback for some variants?
            rotary_emb = model_module.rotary_emb(inputs_embeds, position_ids)
        else:
            raise RuntimeError("Cannot find rotary_emb in model_module")
        
        # Create Attention Mask (Hybrid: Causal for Text, optional Bidirectional for Image tokens)
        bsz, seq_len = input_ids.shape

        # 1. Start with a cached standard Causal Mask (Upper is -inf)
        cache_key = (inputs_embeds.device, inputs_embeds.dtype, int(seq_len))
        base_mask = self._causal_mask_cache.get(cache_key)
        if base_mask is None:
            base_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype) * float('-inf'),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
            self._causal_mask_cache[cache_key] = base_mask
        mask = base_mask.expand(bsz, 1, seq_len, seq_len).clone()  # [B, 1, L, L]

        # 2. Optional: refine mask for Image bidirectional attention.
        # NOTE: Doing anything O(L^2) here can become a bottleneck. We gate this behind config.
        enable_image_bidir_mask = False
        try:
            enable_image_bidir_mask = bool(OmegaConf.select(self.config, "doublevla.enable_image_bidir_mask", default=False))
        except Exception:
            enable_image_bidir_mask = False

        if enable_image_bidir_mask and pixel_values is not None and image_mask is not None:
            # image_mask: [B, L] boolean tensor from get_placeholder_mask
            # Enable bidirectional attention within each contiguous image-token segment.
            for b in range(bsz):
                img_indices = torch.nonzero(image_mask[b], as_tuple=False).flatten()
                if img_indices.numel() <= 1:
                    continue

                diffs = img_indices[1:] - img_indices[:-1]
                split_points = torch.nonzero(diffs > 1, as_tuple=False).flatten()

                # Segment starts
                starts = torch.cat([img_indices[:1], img_indices[1:][diffs > 1]], dim=0)
                # Segment ends (exclusive)
                ends = torch.cat([img_indices[split_points] + 1, img_indices[-1:] + 1], dim=0)

                # Small number of segments; converting these ints is cheap.
                for start, end in zip(starts.tolist(), ends.tolist()):
                    if end - start > 1:
                        mask[b, 0, start:end, start:end] = 0.0

        # Apply padding mask
        if attention_mask is not None:
            # attention_mask is [B, L]
            # Make it [B, 1, 1, L]
            expanded_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype) 
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(inputs_embeds.dtype).min
            mask = mask + expanded_mask

        return inputs_embeds, position_ids, rotary_emb, mask, None


    def _project_qkv(self, attn_module, hidden_states):
        """
        Project hidden states into Q, K, V (and optionally a gate for Qwen3.5).

        Returns:
            Tuple (query_states, key_states, value_states, gate)
            where `gate` is a float tensor for Qwen3.5 gated attention output,
            or None for Qwen2.5-VL / Qwen3-VL.
        """
        bsz, q_len, _ = hidden_states.size()
        gate = None

        if self.is_qwen3_5:
            # Qwen3.5 Attention: q_proj output is 2× head_dim per head (query + gated output gate).
            # Reference: Qwen3_5Attention.forward in modeling_qwen3_5.py
            input_shape = (bsz, q_len)
            hidden_shape = (*input_shape, -1, attn_module.head_dim)

            # Split q_proj output into query and gate halves
            q_raw = attn_module.q_proj(hidden_states).view(*input_shape, -1, attn_module.head_dim * 2)
            query_states_raw, gate_raw = torch.chunk(q_raw, 2, dim=-1)  # each [B, L, n_heads, head_dim]
            gate = gate_raw.reshape(bsz, q_len, -1)  # [B, L, n_heads * head_dim] – applied after o_proj

            # Apply per-head RMSNorm on Q and K (same as Qwen3)
            query_states = attn_module.q_norm(query_states_raw.view(hidden_shape)).transpose(1, 2)
            key_states = attn_module.k_norm(
                attn_module.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        elif hasattr(attn_module, "q_norm"):
            # Qwen3-VL: q_proj has standard head_dim (no gate), but has per-head RMSNorm
            # Qwen3VLTextAttention forward:
            # query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            hidden_shape = (bsz, q_len, -1, attn_module.head_dim)

            query_states = attn_module.q_proj(hidden_states).view(hidden_shape)
            key_states = attn_module.k_proj(hidden_states).view(hidden_shape)
            value_states = attn_module.v_proj(hidden_states).view(hidden_shape)

            query_states = attn_module.q_norm(query_states).transpose(1, 2)
            key_states = attn_module.k_norm(key_states).transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        else:
            # Qwen2.5-VL / Standard: no per-head norm, no gate
            query_states = attn_module.q_proj(hidden_states).view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)
            key_states = attn_module.k_proj(hidden_states).view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)
            value_states = attn_module.v_proj(hidden_states).view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)

        return query_states, key_states, value_states, gate

    def _fix_truncated_vlm_inputs(self, inputs: dict) -> dict:
        """
        Fix image token/feature mismatch caused by input_ids truncation in VLM data.
        
        The VLM dataloader collator truncates input_ids to model_max_length, but does NOT
        correspondingly truncate pixel_values and image_grid_thw. This causes a mismatch
        where get_placeholder_mask finds fewer <|image_pad|> tokens than get_image_features
        produces features, leading to ValueError.
        
        Fix strategy: keep only COMPLETE images whose tokens are fully present in input_ids.
        Remove pixel data for truncated images and replace residual partial image tokens with pad.
        """
        input_ids = inputs.get('input_ids')
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')
        
        if input_ids is None or pixel_values is None or image_grid_thw is None:
            return inputs  # No image data, nothing to fix
        
        # Ensure pad_token_id is valid
        pad_token_id = self.trainable_vlm.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            # Fallback: use eos_token_id or a safe default
            pad_token_id = self.trainable_vlm.processor.tokenizer.eos_token_id
            if pad_token_id is None:
                logger.warning("No valid pad_token_id found, using 0 as fallback")
                pad_token_id = 0
        
        # Get image token ID from model config
        image_token_id = getattr(self.trainable_vlm.model.config, 'image_token_id', 151655)
        
        # Count actual image tokens in (potentially truncated) input_ids
        n_image_tokens = (input_ids == image_token_id).sum().item()
        
        # Get merge_size for calculating expected token counts
        merge_size = 2  # Default for Qwen2.5/3-VL
        try:
            merge_size = self.trainable_vlm.model.config.vision_config.spatial_merge_size
        except AttributeError:
            pass
        
        # Calculate expected tokens (after spatial merge) and raw patches per image
        tokens_per_image = [(g.prod().item() // (merge_size ** 2)) for g in image_grid_thw]
        patches_per_image = [g.prod().item() for g in image_grid_thw]
        total_expected_tokens = sum(tokens_per_image)
        
        if n_image_tokens >= total_expected_tokens:
            return inputs  # No truncation detected, all good
        
        # ---- Truncation detected: fix the mismatch ----
        logger.warning(
            f"VLM input truncation detected: {n_image_tokens} image tokens in input_ids, "
            f"but {total_expected_tokens} expected from {len(image_grid_thw)} images."
        )
        
        fixed = dict(inputs)  # Shallow copy to avoid mutating original
        
        # Determine how many COMPLETE images remain in the truncated input_ids
        cumsum_tokens = 0
        cumsum_patches = 0
        keep_count = 0
        for tok_count, patch_count in zip(tokens_per_image, patches_per_image):
            if cumsum_tokens + tok_count <= n_image_tokens:
                cumsum_tokens += tok_count
                cumsum_patches += patch_count
                keep_count += 1
            else:
                break
        
        logger.warning(
            f"Keeping {keep_count}/{len(image_grid_thw)} complete images ({cumsum_tokens} tokens). "
            f"pixel_values shape: {pixel_values.shape}, cumsum_patches: {cumsum_patches}"
        )
        
        if keep_count == 0:
            # No complete images remain - treat as text-only sample
            fixed['pixel_values'] = None
            fixed['image_grid_thw'] = None
            # Replace all image tokens with pad
            fixed_ids = input_ids.clone()
            fixed_ids[fixed_ids == image_token_id] = pad_token_id
            fixed['input_ids'] = fixed_ids
            fixed['attention_mask'] = fixed_ids.ne(pad_token_id)
        else:
            # Safety check: ensure cumsum_patches doesn't exceed pixel_values size
            actual_patches = pixel_values.shape[0]
            if cumsum_patches > actual_patches:
                logger.error(
                    f"Patch count mismatch: cumsum_patches={cumsum_patches} > actual={actual_patches}. "
                    f"Using min(cumsum_patches, actual_patches)."
                )
                cumsum_patches = actual_patches
                # Recalculate keep_count based on actual patches
                recalc_patches = 0
                keep_count = 0
                for patch_count in patches_per_image:
                    if recalc_patches + patch_count <= cumsum_patches:
                        recalc_patches += patch_count
                        keep_count += 1
                    else:
                        break
                cumsum_patches = recalc_patches
            
            # Keep only complete images' pixel data
            fixed['pixel_values'] = pixel_values[:cumsum_patches]
            fixed['image_grid_thw'] = image_grid_thw[:keep_count]
            
            # Remove partial image tokens (excess beyond last complete image)
            excess = n_image_tokens - cumsum_tokens
            if excess > 0:
                fixed_ids = input_ids.clone()
                # Find all image token positions
                image_token_mask = (fixed_ids == image_token_id)
                
                # Safety: check we have enough tokens to replace
                if image_token_mask.sum().item() < excess:
                    logger.warning(
                        f"Excess tokens ({excess}) > actual image tokens ({image_token_mask.sum().item()}). "
                        f"Replacing all image tokens."
                    )
                    fixed_ids[image_token_mask] = pad_token_id
                else:
                    # Find positions and replace last `excess` tokens
                    # Handle both 1D and 2D input_ids
                    if fixed_ids.dim() == 1:
                        positions = image_token_mask.nonzero(as_tuple=False).squeeze(-1)  # [N]
                        if positions.numel() >= excess:
                            fixed_ids[positions[-excess:]] = pad_token_id
                    else:  # 2D case [B, L]
                        positions = image_token_mask.nonzero(as_tuple=False)  # [N, 2]
                        if positions.shape[0] >= excess:
                            for pos in positions[-excess:]:
                                fixed_ids[pos[0], pos[1]] = pad_token_id
                
                fixed['input_ids'] = fixed_ids
                fixed['attention_mask'] = fixed_ids.ne(pad_token_id)
        
        # Remove pre-computed position_ids so the model recomputes them
        # (they were based on the original untruncated image_grid_thw)
        fixed.pop('position_ids', None)
        
        return fixed

    # ---------------------------------------------------------------------
    # LLR Computation Methods (from LangForce)
    # ---------------------------------------------------------------------
    def _ensure_im_end_id(self, tokenizer):
        """Cache the <|im_end|> token id"""
        if self._im_end_id is None:
            self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _find_last_pos(self, seq_1d: torch.Tensor, token_id: int) -> int:
        """Find the last position of a token in a 1D sequence"""
        idx = (seq_1d == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(idx[-1].item())

    def _find_first_pos_after(self, seq_1d: torch.Tensor, token_id: int, start: int) -> int:
        """Find the first position of a token after a given start position"""
        if start < 0:
            start = 0
        sub = seq_1d[start:]
        idx = (sub == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(start + idx[0].item())

    def _get_action_block_start(self, input_ids_1d: torch.Tensor) -> int:
        """Find the start position of the action token block"""
        action_token_id = self.action_token_id
        action_query_token_num = self.action_query_token_num

        pos = (input_ids_1d == int(action_token_id)).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            return -1

        start = int(pos[0].item())
        end = start + action_query_token_num
        if end > input_ids_1d.shape[0]:
            return -1
        # Verify contiguous block
        if int(input_ids_1d[end - 1].item()) != int(action_token_id):
            return -1
        return start

    def _token_nll_span(
        self,
        logits_1d: torch.Tensor,      # [S, V]
        input_ids_1d: torch.Tensor,   # [S]
        start: int,
        end: int,
        ignore_ids: Optional[Set[int]] = None,
    ):
        """
        Return (nll_vec, target_ids_vec) for tokens in [start,end),
        using next-token alignment:
          token at position j is scored by logits[j-1] (requires j>0).
        """
        if end <= start:
            return None, None
        S = int(input_ids_1d.shape[0])
        start = max(0, int(start))
        end = min(S, int(end))
        if end <= start:
            return None, None

        j = torch.arange(start, end, device=input_ids_1d.device, dtype=torch.long)
        j = j[j > 0]
        if j.numel() == 0:
            return None, None

        targets = input_ids_1d[j].long()

        if ignore_ids is not None and len(ignore_ids) > 0:
            keep = torch.ones_like(targets, dtype=torch.bool)
            for tid in ignore_ids:
                keep &= (targets != int(tid))
            j = j[keep]
            if j.numel() == 0:
                return None, None
            targets = input_ids_1d[j].long()

        pred_pos = j - 1
        pred_logits = logits_1d[pred_pos].float()  # [T, V]
        nll = F.cross_entropy(pred_logits, targets, reduction="none")  # [T]
        return nll, targets

    def _compute_language_llr_from_boundaries(
        self,
        prior_logits: torch.Tensor,            # [B, S, V]
        posterior_logits: torch.Tensor,        # [B, S, V] (detached)
        prior_input_ids: torch.Tensor,         # [B, S]
        posterior_input_ids: torch.Tensor,     # [B, S]
        prior_action_starts: torch.Tensor,     # [B]
        posterior_action_starts: torch.Tensor, # [B]
        use_hard_token_llr: bool = True,
        hard_token_k: int = 16,
        assert_lang_span_match: bool = True,
    ) -> torch.Tensor:
        """
        Compute LLR: log p(L|V,A_prior) - log p(L|V)
        
        Prior language span: [action_end : im_end) - language AFTER action tokens
        Posterior language span: [last(vision_end)+1 : action_start) - language BEFORE action tokens
        """
        tokenizer = self.trainable_vlm.processor.tokenizer
        self._ensure_im_end_id(tokenizer)

        pad_id = tokenizer.pad_token_id
        ignore_ids: Set[int] = set()
        if pad_id is not None:
            ignore_ids.add(int(pad_id))
        ignore_ids.add(int(IMAGE_TOKEN_INDEX))
        ignore_ids.add(int(VIDEO_TOKEN_INDEX))
        ignore_ids.add(int(VISION_START_TOKEN_INDEX))
        ignore_ids.add(int(VISION_END_TOKEN_INDEX))
        ignore_ids.add(int(IM_START_TOKEN_INDEX))
        ignore_ids.add(int(IM_END_TOKEN_INDEX))

        B = int(prior_input_ids.shape[0])
        K = self.action_query_token_num

        llr_vals = []
        post_nll_means = []

        for b in range(B):
            ids_prior = prior_input_ids[b]
            ids_post = posterior_input_ids[b]

            a_start_prior = int(prior_action_starts[b].item())
            a_start_post = int(posterior_action_starts[b].item())

            # ===== Prior language span: [action_end : im_end) =====
            # In prior mode (A + L), language comes AFTER action tokens
            lang_start_prior = a_start_prior + K
            if lang_start_prior >= ids_prior.shape[0]:
                continue
            im_end = self._find_first_pos_after(ids_prior, self._im_end_id, lang_start_prior)
            lang_end_prior = im_end if im_end != -1 else int(ids_prior.shape[0])
            if lang_end_prior <= lang_start_prior:
                continue

            # ===== Posterior language span: [last(vision_end)+1 : action_start) =====
            # In posterior mode (L + A), language comes BEFORE action tokens
            v_end_post = self._find_last_pos(ids_post, VISION_END_TOKEN_INDEX)
            if v_end_post == -1:
                continue
            lang_start_post = v_end_post + 1
            lang_end_post = a_start_post
            if lang_end_post <= lang_start_post:
                continue

            # ===== (1) Strict assertion: token-level equality =====
            if self.training and assert_lang_span_match:
                prior_span_ids = ids_prior[lang_start_prior:lang_end_prior]
                post_span_ids = ids_post[lang_start_post:lang_end_post]

                if (prior_span_ids.numel() != post_span_ids.numel()) or (not torch.equal(prior_span_ids, post_span_ids)):
                    # Decode for human-readable debugging
                    prior_text = tokenizer.decode(prior_span_ids.tolist())
                    post_text = tokenizer.decode(post_span_ids.tolist())

                    raise AssertionError(
                        "\n[DoubleVLA+LangForce] Language span mismatch detected!\n"
                        f"Sample b={b}\n"
                        f"PRIOR span idx: [{lang_start_prior}:{lang_end_prior}]  (len={prior_span_ids.numel()})\n"
                        f"POST  span idx: [{lang_start_post}:{lang_end_post}]  (len={post_span_ids.numel()})\n"
                        f"PRIOR span: {repr(prior_text)}\n"
                        f"POST  span: {repr(post_text)}\n"
                        f"PRIOR token ids (first 50): {prior_span_ids[:50].tolist()}\n"
                        f"POST  token ids (first 50): {post_span_ids[:50].tolist()}\n"
                        "This indicates your boundary-based language extraction is inconsistent."
                    )

            # ===== (2) Compute NLL for both spans =====
            nll_prior, tok_prior = self._token_nll_span(
                logits_1d=prior_logits[b],
                input_ids_1d=ids_prior,
                start=lang_start_prior,
                end=lang_end_prior,
                ignore_ids=ignore_ids,
            )
            nll_post, tok_post = self._token_nll_span(
                logits_1d=posterior_logits[b],
                input_ids_1d=ids_post,
                start=lang_start_post,
                end=lang_end_post,
                ignore_ids=ignore_ids,
            )
            if nll_prior is None or nll_post is None:
                continue

            # Record posterior NLL mean for potential gating
            post_nll_mean = nll_post.mean().detach()
            post_nll_means.append(post_nll_mean)

            # LLR = log p(L|V,A) - log p(L|V) = (-nll_prior) - (-nll_post) = nll_post - nll_prior
            if use_hard_token_llr:
                # Require same target token sequence
                if tok_prior is None or tok_post is None or tok_prior.shape != tok_post.shape or (not torch.equal(tok_prior, tok_post)):
                    # Fallback: use mean
                    llr = (nll_post.mean() - nll_prior.mean())
                else:
                    k = min(hard_token_k, int(nll_post.numel()))
                    if k <= 0:
                        continue
                    # Select top-k hardest tokens under posterior
                    idx = torch.topk(nll_post.detach(), k=k, largest=True).indices
                    llr = (nll_post[idx] - nll_prior[idx]).mean()
            else:
                llr = (nll_post.mean() - nll_prior.mean())

            llr_vals.append(llr)

        if len(llr_vals) == 0:
            return torch.tensor(0.0, device=prior_logits.device, dtype=torch.float32)

        llr_vals_t = torch.stack(llr_vals).float()
        return llr_vals_t.mean()


@FRAMEWORK_REGISTRY.register("PhysBrainVLA")
class PhysBrainVLA(baseframework):
    """
    PhysBrainVLA + LangForce Framework
    
    Combined architecture:
    - PhysBrainVLA: dual-brain architecture (frozen left brain + trainable right brain), connected via MoT
    - LangForce: Bayesian decomposition with LLR regularization to prevent visual shortcuts
    
    Training flow:
    1. Prior Forward: left brain (V+L) + right brain (V+A+L) -> p(a|v), log p(L|V,A)
    2. Posterior Forward: left brain (V+L) + right brain (V+L+A) -> π(a|v,l), log p(L|V)
    3. LLR Loss: maximize log p(L|V,A) - sg(log p(L|V))
    4. Action Loss: Flow-matching loss for both branches
    
    Components:
      - Double Qwen2.5/3 VL interface (MoT: Frozen + Trainable)
      - Flow-matching action head for VLA tasks
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.vla_config = DoubleVLAConfig.from_dict(config.get("doublevla", {}))
        
        # Use our new MoT interface
        self.qwen_vl_interface = MoT_Qwen_VL_Interface(config=self.config, vla_config=self.vla_config)
        
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # ===== LangForce Configuration =====
        # Enable/disable LangForce mode
        self.use_langforce = bool(self.config.framework.get("use_langforce", True))
        
        # LLR loss weight (maximize LLR via -kl_weight * kl_loss)
        self.kl_weight = float(self.config.framework.get("kl_weight", 0.1))
        
        # Prior action loss weight
        self.prior_loss_weight = float(self.config.framework.get("prior_loss_weight", 0.3))
        
        # Hard-token LLR configuration
        self.use_hard_token_llr = bool(self.config.framework.get("use_hard_token_llr", True))
        self.hard_token_k = int(self.config.framework.get("hard_token_k", 16))
        
        # Strict span assertion during training
        self.assert_lang_span_match = bool(self.config.framework.get("assert_lang_span_match", True))
        
        # Detach prior condition (protect backbone from vision-only drift)
        self.detach_prior_cond = bool(self.config.framework.get("detach_prior_cond", True))
        
        if self.use_langforce:
            logger.info(f"🟢 LangForce enabled: kl_weight={self.kl_weight}, prior_loss_weight={self.prior_loss_weight}")
            logger.info(f"🟢 Hard-token LLR: {self.use_hard_token_llr}, k={self.hard_token_k}")

    def __call__(self, *args, **kwargs):
        """
        Callable interface that routes to appropriate mode for co-training.
        
        Usage:
            # VLA mode (action prediction training)
            model(examples=[...])  # examples contain 'action' field
            
            # VLM mode (VQA training)
            model(input_ids=..., attention_mask=..., labels=..., pixel_values=...)
        
        Returns:
            VLA mode: dict with 'action_loss'
            VLM mode: Qwen output with 'loss' and 'logits'
        """
        # Check if this is a VLA call (has 'examples' or first arg is list of dicts with 'action')
        if 'examples' in kwargs:
            # VLA mode - call forward
            return self.forward(**kwargs)
        elif args and isinstance(args[0], list) and len(args[0]) > 0 and isinstance(args[0][0], dict):
            # VLA mode - examples passed as first positional argument
            if 'action' in args[0][0]:
                return self.forward(examples=args[0], **kwargs)
            # If no 'action' field, might be inference mode, still use forward
            return self.forward(examples=args[0], **kwargs)
        else:
            # VLM mode - forward tokenizer outputs to MoT interface
            # This is for VQA/language modeling training
            logger.info("🔵 VLM mode: forwarding to MoT interface for VQA training")
            return self.qwen_vl_interface.forward(
                is_vla=False,
                frozen_inputs=kwargs,  # VLM mode doesn't need frozen VLM
                trainable_inputs=kwargs,
                state_tensor=None,
                embodiment_id_tensor=None,
            )

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference uses Posterior branch: (V + L + action_query)
        
        Steps:
          1. Resize images to training resolution (if specified)
                    2. Encode with PhysBrainVLA (left brain V+L, right brain V+L+A)
          3. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        embodiment_id = [example["embodiment_id"] for example in examples] if "embodiment_id" in examples[0] else None  # [B]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: Build separate inputs (posterior mode for inference)
        frozen_inputs, trainable_inputs, state_tensor, embodiment_id_tensor = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            state=state,
            embodiment_id=embodiment_id,
            mode='posterior',  # Use posterior mode for inference
        )
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs, _ = self.qwen_vl_interface(
                is_vla=True,
                frozen_inputs=frozen_inputs,
                trainable_inputs=trainable_inputs,
                state_tensor=state_tensor,
                embodiment_id_tensor=embodiment_id_tensor,
                selected_layer=self.vla_config.hidden_states_selected_layer,  # Select which layer to extract
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # selected_hidden_state: [B, N, H]
            selected_hidden_state = qwenvl_outputs.hidden_states[0]

        state = torch.from_numpy(np.array(state)).to(selected_hidden_state.device, dtype=selected_hidden_state.dtype) if state is not None else None
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(selected_hidden_state, state)  # (B, chunk_len, action_dim)
        
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/SimplerEnv/train_files/starvla_cotrain_oxe_qwen3.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()
    

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.action_model.action_dim = 7
    cfg.framework.action_model.state_dim = 8
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

    model = PhysBrainVLA(cfg)
    # print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    action_dim = cfg.framework.action_model.action_dim
    state_dim = cfg.framework.action_model.state_dim
    
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, action_dim)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        "state" : np.random.uniform(-1, 1, size=(1, state_dim)).astype(np.float16), # chunk, state_dim
        "embodiment_id": 0,
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "True"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # 
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        print(batch)
        print(batch[0].keys())
        break

    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)

    action = model.predict_action(examples=batch)
    print("Finished")
