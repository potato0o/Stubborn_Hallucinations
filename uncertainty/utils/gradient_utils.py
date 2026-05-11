import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def _count_target_tokens(labels: torch.Tensor) -> int:
    """Count valid (non -100) target tokens."""
    return int((labels != -100).sum().item())


def get_gradients_fixed(
    model,
    tokenizer,
    input_text: str,
    target_ids: torch.Tensor,
    layer_type: str = "lm_head",
    device: str = "cuda",
    target_mask: torch.Tensor = None,
):
    """
    Compute gradients of the negative log-likelihood of FIXED target tokens
    w.r.t. a chosen model component.

    This implementation is:
    - Tokenization-drift free
    - BOS-safe
    - Masking-correct
    - Deterministic across perturbations

    Args:
        target_ids (Tensor): Shape (1, T) or (T,). These MUST be the exact token
                             IDs whose likelihood is evaluated.
        target_mask (Tensor, optional): Shape (T,). Binary mask where 1 = include in loss,
                                        0 = exclude. If None, all target tokens contribute.
    """

    model.eval()
    model.zero_grad(set_to_none=True)

    # ---------------------------------------------------------
    # 1. Tokenize prompt ONLY
    # ---------------------------------------------------------
    input_tokens = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=2048  # Add truncation to prevent extremely long sequences
    ).to(device)
    input_ids = input_tokens.input_ids  # (1, P)

    # ---------------------------------------------------------
    # 2. Prepare fixed target IDs
    # ---------------------------------------------------------
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)

    target_ids = target_ids.to(device)  # (1, T)

    # IMPORTANT: target_ids must NOT include BOS
    if tokenizer.bos_token_id is not None:
        if target_ids.shape[1] > 0 and target_ids[0, 0].item() == tokenizer.bos_token_id:
            raise ValueError(
                "target_ids should NOT include BOS token. "
                "They must represent only the answer tokens."
            )

    # ---------------------------------------------------------
    # 3. Concatenate prompt + fixed target
    # ---------------------------------------------------------
    full_ids = torch.cat([input_ids, target_ids], dim=1)  # (1, P+T)
    
    # Validate total length
    if full_ids.shape[1] > 4096:  # Prevent extremely long sequences
        raise ValueError(f"Sequence too long: {full_ids.shape[1]} tokens. Maximum is 4096.")

    # ---------------------------------------------------------
    # 4. Create labels (mask prompt only)
    # ---------------------------------------------------------
    labels = full_ids.clone()
    labels[:, :input_ids.shape[1]] = -100  # Only target contributes to loss

    # ---------------------------------------------------------
    # 5. Apply target mask if provided (key phrase masking)
    # ---------------------------------------------------------
    if target_mask is not None:
        if target_mask.dim() == 1:
            target_mask = target_mask.unsqueeze(0)  # (1, T)
        
        target_mask = target_mask.to(device)
        
        # Ensure mask length matches target length
        target_len = target_ids.shape[1]
        if target_mask.shape[1] != target_len:
            raise ValueError(
                f"Mask length {target_mask.shape[1]} doesn't match target length {target_len}"
            )
        
        # Apply mask: set non-masked target tokens to -100
        # Mask layout: prompt_len positions already -100, then apply mask to target positions
        target_start_idx = input_ids.shape[1]
        for i in range(target_len):
            if target_mask[0, i] == 0:
                labels[0, target_start_idx + i] = -100

    num_target_tokens = _count_target_tokens(labels)
    if num_target_tokens == 0:
        raise ValueError("No valid target tokens for gradient computation (all masked out).")

    # ---------------------------------------------------------
    # 6. Select target layer
    # ---------------------------------------------------------
    if layer_type == "lm_head":
        target_layer = (
            model.lm_head
            if hasattr(model, "lm_head")
            else model.get_output_embeddings()
        )

    elif layer_type == "layer_norm":
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            target_layer = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            target_layer = model.transformer.ln_f
        elif hasattr(model, "norm"):
            target_layer = model.norm
        else:
            raise ValueError("Final LayerNorm not found.")

    elif layer_type in ["last_block", "last_transformer_block"]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            target_layer = model.model.layers[-1]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            target_layer = model.transformer.h[-1]
        else:
            raise ValueError("Last transformer block not found.")

    elif layer_type in ["middle_block", "middle_transformer_block"]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            middle_idx = len(model.model.layers) // 2
            target_layer = model.model.layers[middle_idx]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            middle_idx = len(model.transformer.h) // 2
            target_layer = model.transformer.h[middle_idx]
        else:
            raise ValueError("Middle transformer block not found.")

    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

    # ---------------------------------------------------------
    # 7. Enable gradients ONLY for target layer
    # ---------------------------------------------------------
    params = [p for p in target_layer.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("Target layer has no trainable parameters.")

    # ---------------------------------------------------------
    # 8. Forward + backward with defensive error handling
    # ---------------------------------------------------------
    try:
        # Synchronize CUDA to ensure previous operations complete
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Get logits without using the default loss
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)
        
        # Validate output shapes
        if logits.shape[1] != full_ids.shape[1]:
            raise ValueError(f"Output length mismatch: logits {logits.shape[1]} vs input {full_ids.shape[1]}")
        
        # Manually compute loss with reduction='sum' to preserve total error magnitude
        # This prevents gradient shrinkage for long answers
        shift_logits = logits[:, :-1, :].contiguous()  # Remove last position
        shift_labels = labels[:, 1:].contiguous()  # Remove first position (BOS)
        
        # Compute cross-entropy with sum reduction
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',  # Sum instead of mean - measures total error energy
            ignore_index=-100
        )

        if not torch.isfinite(loss):
            raise ValueError("Loss is NaN or Inf.")

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            create_graph=False,
        )
        
        # Synchronize again after backward
        if device == "cuda":
            torch.cuda.synchronize()

    except RuntimeError as e:
        # Catch CUDA errors or memory access violations
        import logging
        logging.error(f"Runtime error during forward/backward: {e}")
        # Clean up and return None to signal failure
        model.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.empty_cache()
        return None

    # ---------------------------------------------------------
    # 9. Flatten gradient vector (NO extra normalization)
    # ---------------------------------------------------------
    grad_vec = torch.cat([g.reshape(-1) for g in grads])

    return grad_vec


def get_gradients_with_embedding_noise(
    model,
    tokenizer,
    input_text: str,
    target_ids: torch.Tensor,
    epsilon: float = 0.01,
    layer_type: str = "lm_head",
    device: str = "cuda",
    target_mask: torch.Tensor = None,
):
    """
    Compute gradients with Gaussian noise injected into input embeddings.
    
    This implements embedding-level perturbation: instead of modifying the text,
    we add small Gaussian noise (ε) directly to the input embeddings of the clean query.
    This creates a continuous perturbation in the embedding space.

    Args:
        model: The language model
        tokenizer: The tokenizer
        input_text (str): The clean input text (prompt)
        target_ids (Tensor): Shape (1, T) or (T,). The exact token IDs for the answer.
        epsilon (float): Standard deviation of the Gaussian noise to inject (default: 0.01)
        layer_type (str): Which layer's gradients to compute
        device (str): Device to use
        target_mask (Tensor, optional): Binary mask for target tokens

    Returns:
        torch.Tensor: Flattened gradient vector, or None if computation fails
    """

    model.eval()
    model.zero_grad(set_to_none=True)

    # ---------------------------------------------------------
    # 1. Tokenize prompt ONLY
    # ---------------------------------------------------------
    input_tokens = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=2048
    ).to(device)
    input_ids = input_tokens.input_ids  # (1, P)

    # ---------------------------------------------------------
    # 2. Prepare fixed target IDs
    # ---------------------------------------------------------
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)

    target_ids = target_ids.to(device)  # (1, T)

    # IMPORTANT: target_ids must NOT include BOS
    if tokenizer.bos_token_id is not None:
        if target_ids.shape[1] > 0 and target_ids[0, 0].item() == tokenizer.bos_token_id:
            raise ValueError(
                "target_ids should NOT include BOS token. "
                "They must represent only the answer tokens."
            )

    # ---------------------------------------------------------
    # 3. Concatenate prompt + fixed target
    # ---------------------------------------------------------
    full_ids = torch.cat([input_ids, target_ids], dim=1)  # (1, P+T)
    
    # Validate total length
    if full_ids.shape[1] > 4096:
        raise ValueError(f"Sequence too long: {full_ids.shape[1]} tokens. Maximum is 4096.")

    # ---------------------------------------------------------
    # 4. Create labels (mask prompt only)
    # ---------------------------------------------------------
    labels = full_ids.clone()
    labels[:, :input_ids.shape[1]] = -100  # Only target contributes to loss

    # ---------------------------------------------------------
    # 5. Apply target mask if provided (key phrase masking)
    # ---------------------------------------------------------
    if target_mask is not None:
        if target_mask.dim() == 1:
            target_mask = target_mask.unsqueeze(0)
        
        target_mask = target_mask.to(device)
        
        target_len = target_ids.shape[1]
        if target_mask.shape[1] != target_len:
            raise ValueError(
                f"Mask length {target_mask.shape[1]} doesn't match target length {target_len}"
            )
        
        target_start_idx = input_ids.shape[1]
        for i in range(target_len):
            if target_mask[0, i] == 0:
                labels[0, target_start_idx + i] = -100

    num_target_tokens = _count_target_tokens(labels)
    if num_target_tokens == 0:
        raise ValueError("No valid target tokens for gradient computation (all masked out).")

    # ---------------------------------------------------------
    # 6. Get input embeddings and add Gaussian noise
    # ---------------------------------------------------------
    # Get the embedding layer
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_layer = model.model.embed_tokens
    elif hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        raise ValueError("Could not find input embedding layer.")

    # Get clean embeddings
    with torch.no_grad():
        clean_embeds = embed_layer(full_ids)  # (1, seq_len, hidden_dim)
    
    # Add Gaussian noise ONLY to the input (prompt) portion, not the target
    # This simulates perturbation of the query while keeping the answer fixed
    noisy_embeds = clean_embeds.clone()
    prompt_len = input_ids.shape[1]
    
    # Generate and add noise to prompt embeddings
    noise = torch.randn_like(clean_embeds[:, :prompt_len, :]) * epsilon
    noisy_embeds[:, :prompt_len, :] = clean_embeds[:, :prompt_len, :] + noise
    
    # Detach and re-enable gradients for the forward pass
    noisy_embeds = noisy_embeds.detach().requires_grad_(False)

    # ---------------------------------------------------------
    # 7. Select target layer (same as get_gradients_fixed)
    # ---------------------------------------------------------
    if layer_type == "lm_head":
        target_layer = (
            model.lm_head
            if hasattr(model, "lm_head")
            else model.get_output_embeddings()
        )
    elif layer_type == "layer_norm":
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            target_layer = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            target_layer = model.transformer.ln_f
        elif hasattr(model, "norm"):
            target_layer = model.norm
        else:
            raise ValueError("Final LayerNorm not found.")
    elif layer_type in ["last_block", "last_transformer_block"]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            target_layer = model.model.layers[-1]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            target_layer = model.transformer.h[-1]
        else:
            raise ValueError("Last transformer block not found.")
    elif layer_type in ["middle_block", "middle_transformer_block"]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            middle_idx = len(model.model.layers) // 2
            target_layer = model.model.layers[middle_idx]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            middle_idx = len(model.transformer.h) // 2
            target_layer = model.transformer.h[middle_idx]
        else:
            raise ValueError("Middle transformer block not found.")
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

    # ---------------------------------------------------------
    # 8. Enable gradients ONLY for target layer
    # ---------------------------------------------------------
    params = [p for p in target_layer.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("Target layer has no trainable parameters.")

    # ---------------------------------------------------------
    # 9. Forward + backward with noisy embeddings
    # ---------------------------------------------------------
    try:
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Forward pass with noisy embeddings (bypass the embedding layer)
        outputs = model(inputs_embeds=noisy_embeds)
        logits = outputs.logits  # (1, seq_len, vocab_size)
        
        if logits.shape[1] != full_ids.shape[1]:
            raise ValueError(f"Output length mismatch: logits {logits.shape[1]} vs input {full_ids.shape[1]}")
        
        # Manually compute loss with reduction='sum'
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
            ignore_index=-100
        )

        if not torch.isfinite(loss):
            raise ValueError("Loss is NaN or Inf.")

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            create_graph=False,
        )
        
        if device == "cuda":
            torch.cuda.synchronize()

    except RuntimeError as e:
        import logging
        logging.error(f"Runtime error during forward/backward with noisy embeddings: {e}")
        model.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.empty_cache()
        return None

    # ---------------------------------------------------------
    # 10. Flatten gradient vector
    # ---------------------------------------------------------
    grad_vec = torch.cat([g.reshape(-1) for g in grads])

    return grad_vec


def calculate_hallucination_score(
    grad_clean: torch.Tensor,
    grad_perturbed: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Hallucination score:
        H = ||g_clean|| * (1 - CosSim(g_clean, g_pert))

    Captures:
    - Epistemic weakness (||g_clean||)
    - Instability under perturbation
    """

    if grad_clean.shape != grad_perturbed.shape:
        raise ValueError("Gradient shapes must match.")

    if grad_clean.device != grad_perturbed.device:
        grad_perturbed = grad_perturbed.to(grad_clean.device)

    # Epistemic term
    epistemic = torch.norm(grad_clean, p=2)

    # Directional instability
    cos_sim = F.cosine_similarity(
        grad_clean.unsqueeze(0),
        grad_perturbed.unsqueeze(0),
        dim=1,
        eps=eps,
    ).clamp(-1.0, 1.0)

    instability = 1.0 - cos_sim.item()

    return epistemic.item() * instability


def get_gradients_with_mc_dropout(
    model,
    tokenizer,
    input_text: str,
    target_ids: torch.Tensor,
    layer_type: str = "lm_head",
    device: str = "cuda",
    target_mask: torch.Tensor = None,
    dropout_rate: float = None,
):
    """
    Compute gradients with MC Dropout enabled (model in train mode).
    
    This implements weight perturbation via stochastic dropout masks.
    Instead of modifying the input (text or embeddings), dropout randomly
    zeros out activations during the forward pass, effectively perturbing
    the model's computation graph.

    Args:
        model: The language model
        tokenizer: The tokenizer
        input_text (str): The input text (prompt)
        target_ids (Tensor): Shape (1, T) or (T,). The exact token IDs for the answer.
        layer_type (str): Which layer's gradients to compute
        device (str): Device to use
        target_mask (Tensor, optional): Binary mask for target tokens
        dropout_rate (float, optional): If provided, temporarily sets dropout rate.
                                        If None, uses the model's existing dropout configuration.

    Returns:
        torch.Tensor: Flattened gradient vector, or None if computation fails
    """

    # Store original dropout rates to restore later
    original_dropout_rates = {}
    
    # Optionally modify dropout rates
    if dropout_rate is not None:
        # Common attribute names for dropout in transformer models
        dropout_attrs = [
            'hidden_dropout_prob',
            'attention_probs_dropout_prob', 
            'resid_pdrop',
            'embd_pdrop',
            'attn_pdrop',
            'dropout',
        ]
        
        if hasattr(model, 'config'):
            for attr in dropout_attrs:
                if hasattr(model.config, attr):
                    original_dropout_rates[attr] = getattr(model.config, attr)
                    setattr(model.config, attr, dropout_rate)

    # Key difference: model.train() to enable dropout
    model.train()
    model.zero_grad(set_to_none=True)

    # ---------------------------------------------------------
    # 1. Tokenize prompt ONLY
    # ---------------------------------------------------------
    input_tokens = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=2048
    ).to(device)
    input_ids = input_tokens.input_ids  # (1, P)

    # ---------------------------------------------------------
    # 2. Prepare fixed target IDs
    # ---------------------------------------------------------
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)

    target_ids = target_ids.to(device)  # (1, T)

    # IMPORTANT: target_ids must NOT include BOS
    if tokenizer.bos_token_id is not None:
        if target_ids.shape[1] > 0 and target_ids[0, 0].item() == tokenizer.bos_token_id:
            raise ValueError(
                "target_ids should NOT include BOS token. "
                "They must represent only the answer tokens."
            )

    # ---------------------------------------------------------
    # 3. Concatenate prompt + fixed target
    # ---------------------------------------------------------
    full_ids = torch.cat([input_ids, target_ids], dim=1)  # (1, P+T)
    
    if full_ids.shape[1] > 4096:
        raise ValueError(f"Sequence too long: {full_ids.shape[1]} tokens. Maximum is 4096.")

    # ---------------------------------------------------------
    # 4. Create labels (mask prompt only)
    # ---------------------------------------------------------
    labels = full_ids.clone()
    labels[:, :input_ids.shape[1]] = -100  # Only target contributes to loss

    # ---------------------------------------------------------
    # 5. Apply target mask if provided
    # ---------------------------------------------------------
    if target_mask is not None:
        if target_mask.dim() == 1:
            target_mask = target_mask.unsqueeze(0)
        
        target_mask = target_mask.to(device)
        
        target_len = target_ids.shape[1]
        if target_mask.shape[1] != target_len:
            raise ValueError(
                f"Mask length {target_mask.shape[1]} doesn't match target length {target_len}"
            )
        
        target_start_idx = input_ids.shape[1]
        for i in range(target_len):
            if target_mask[0, i] == 0:
                labels[0, target_start_idx + i] = -100

    num_target_tokens = _count_target_tokens(labels)
    if num_target_tokens == 0:
        raise ValueError("No valid target tokens for gradient computation (all masked out).")

    # ---------------------------------------------------------
    # 6. Select target layer
    # ---------------------------------------------------------
    if layer_type == "lm_head":
        target_layer = (
            model.lm_head
            if hasattr(model, "lm_head")
            else model.get_output_embeddings()
        )
    elif layer_type == "layer_norm":
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            target_layer = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            target_layer = model.transformer.ln_f
        elif hasattr(model, "norm"):
            target_layer = model.norm
        else:
            raise ValueError("Final LayerNorm not found.")
    elif layer_type in ["last_block", "last_transformer_block"]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            target_layer = model.model.layers[-1]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            target_layer = model.transformer.h[-1]
        else:
            raise ValueError("Last transformer block not found.")
    elif layer_type in ["middle_block", "middle_transformer_block"]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            middle_idx = len(model.model.layers) // 2
            target_layer = model.model.layers[middle_idx]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            middle_idx = len(model.transformer.h) // 2
            target_layer = model.transformer.h[middle_idx]
        else:
            raise ValueError("Middle transformer block not found.")
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

    # ---------------------------------------------------------
    # 7. Enable gradients ONLY for target layer
    # ---------------------------------------------------------
    params = [p for p in target_layer.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("Target layer has no trainable parameters.")

    # ---------------------------------------------------------
    # 8. Forward + backward with MC Dropout active
    # ---------------------------------------------------------
    try:
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Forward pass with dropout enabled (via model.train())
        outputs = model(full_ids)
        logits = outputs.logits

        if logits.shape[1] != full_ids.shape[1]:
            raise ValueError(f"Output length mismatch: logits {logits.shape[1]} vs input {full_ids.shape[1]}")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
            ignore_index=-100
        )

        if not torch.isfinite(loss):
            raise ValueError("Loss is NaN or Inf.")

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            create_graph=False,
        )
        
        if device == "cuda":
            torch.cuda.synchronize()

    except RuntimeError as e:
        import logging
        logging.error(f"Runtime error during forward/backward with MC Dropout: {e}")
        model.zero_grad(set_to_none=True)
        model.eval()  # Restore eval mode
        # Restore original dropout rates
        if hasattr(model, 'config'):
            for attr, val in original_dropout_rates.items():
                setattr(model.config, attr, val)
        if device == "cuda":
            torch.cuda.empty_cache()
        return None

    finally:
        # Always restore model to eval mode and original dropout rates
        model.eval()
        if hasattr(model, 'config'):
            for attr, val in original_dropout_rates.items():
                setattr(model.config, attr, val)

    # ---------------------------------------------------------
    # 9. Flatten gradient vector
    # ---------------------------------------------------------
    grad_vec = torch.cat([g.reshape(-1) for g in grads])

    return grad_vec

