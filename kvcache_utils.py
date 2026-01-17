import torch
from typing import Tuple

def process_kv_cache(
    past_key_values,
    model,
    system_size: int,
    inst_size: int,
    token_per_frame: int,
    compress_frame_num: int,
    method: str = "uniform",
    tar_ratio: float = 0.5,
    query_ratio: float = 0.25,
    adaptive_pooling: bool = False,
    is_first_block: bool = False,
    is_last_block: bool = False,
    per_frame: bool = False
) -> Tuple:
    """
    Compress KV cache by keeping only specified number of frames in vision tokens
    
    Args:
        past_key_values: Current KV cache (DynamicCache)
        model: The model instance
        system_size: Size of system tokens
        inst_size: Size of instruction tokens
        token_per_frame: Number of tokens per frame
        compress_frame_num: Number of frames to keep after compression
        method: Compression method ('swa', 'uniform', 'tar_val')
        tar_ratio: Ratio for tar vs other methods (used in 'tar_val')
        query_ratio: Ratio of query frames for tar method
        adaptive_pooling: Whether to use adaptive pooling
        is_first_block: Whether this is the first block
        is_last_block: Whether this is the last block
        per_frame: Whether to select complete frames (for val_norm method)
        
    Returns:
        tuple: (compressed_past_key_values, cap_list)
    """
    
    pooling_func_list = [3, 2, 1, -1]
    pool_size_list = [7, 5, 3, 1]

    if compress_frame_num <= 0:
        return past_key_values, None
    
    # Get current sequence length
    current_seq_len = past_key_values.get_seq_length()
    
    # Calculate vision token range
    vision_start = system_size
    vision_end = current_seq_len
    vision_length = vision_end - vision_start
    current_frame_num = vision_length // token_per_frame
    assert vision_length % token_per_frame == 0, "Vision length must be divisible by token_per_frame"
    
    if current_frame_num <= compress_frame_num:
        return past_key_values, None
    
    # Get key and value states for processing
    num_layers = len(model.model.language_model.layers)
    
    # Process each layer
    for layer_idx in range(num_layers):

        # Layer-Adaptive Pooling
        if adaptive_pooling:
            num_layers = len(model.model.language_model.layers)
            idx = layer_idx // max(1, (num_layers // len(pooling_func_list)))
            #idx = layer_idx // (model.config.num_hidden_layers // len(pooling_func_list))
            avg_pooling_nd = pooling_func_list[idx]
            attn_pool_size = pool_size_list[idx]
        else:
            avg_pooling_nd = -1

        key_states = past_key_values.layers[layer_idx].keys
        value_states = past_key_values.layers[layer_idx].values
        bsz, num_heads, q_len, head_dim = key_states.shape
        
        # Extract vision tokens to compress
        key_states_to_compress = key_states[:, :, system_size:, :]
        value_states_to_compress = value_states[:, :, system_size:, :]
        
        # Select compression method
        # Calculate total tokens to keep
        total_tokens_to_keep = compress_frame_num * token_per_frame
        
        if method == "swa":
            # SWA: Keep the most recent tokens
            start_idx = key_states_to_compress.shape[2] - total_tokens_to_keep
            all_indices = torch.arange(start_idx, key_states_to_compress.shape[2], device=key_states.device)
            all_indices = all_indices.unsqueeze(0).expand(num_heads, -1)
            
        elif method == "uniform":
            # Uniform: Sample tokens uniformly across the sequence
            total_tokens = key_states_to_compress.shape[2]
            step = total_tokens / total_tokens_to_keep
            selected_token_indices = [int(i * step) for i in range(total_tokens_to_keep)]
            all_indices = torch.tensor(selected_token_indices, device=key_states.device)
            all_indices = all_indices.unsqueeze(0).expand(num_heads, -1)
                        
        elif method == "infinipot-v":
            # Combined tar + Value norm approach
            attn_budget = round((1 - tar_ratio) * compress_frame_num * token_per_frame)
            
            # tar part
            if tar_ratio > 0:
                tar_budget = compress_frame_num * token_per_frame - attn_budget
                query_frame = int(current_frame_num * query_ratio)
                query_length = int(query_frame * token_per_frame)
                total_length = key_states_to_compress.shape[2]
                
                # Get query and key embeddings
                query_frame_emb = key_states_to_compress[:, :, -query_length:, :]
                query_emb_norm = query_frame_emb / (query_frame_emb.norm(dim=-1, keepdim=True) + 1e-9)
                
                key_frame_emb = key_states_to_compress[:, :, :-query_length, :]
                key_emb_norm = key_frame_emb / (key_frame_emb.norm(dim=-1, keepdim=True) + 1e-9)
                
                # Pix2pix similarity calculation
                query_emb_norm_reshaped = query_emb_norm.reshape(
                    query_emb_norm.shape[0], query_emb_norm.shape[1],
                    query_frame, token_per_frame, query_emb_norm.shape[-1]
                )
                key_emb_norm_reshaped = key_emb_norm.reshape(
                    key_emb_norm.shape[0], key_emb_norm.shape[1],
                    -1, token_per_frame, key_emb_norm.shape[-1]
                )
                
                key_score = -(
                    query_emb_norm_reshaped.unsqueeze(3) *   # [b, h, q, 1, t, d]
                    key_emb_norm_reshaped.unsqueeze(2)       # [b, h, 1, k, t, d]
                ).sum(dim=-1).mean(dim=2).reshape(query_emb_norm.shape[0], query_emb_norm.shape[1], -1)
                
                recent_indices = torch.arange(
                    total_length - query_length, total_length
                ).unsqueeze(0).expand(num_heads, -1).to(key_states_to_compress.device)
                
                selected_indices = key_score.topk(tar_budget - query_length, dim=-1).indices.squeeze(0)
                tar_indices_combined = torch.cat([selected_indices, recent_indices], dim=-1)
                tar_indices, _ = tar_indices_combined.sort(dim=-1)
            else:
                tar_indices = None
            
            # Value norm part
            val_norm_score = value_states_to_compress.norm(dim=-1)[0, :, :]  # [head, seq_len]
            
            # Apply pooling if enabled
            if avg_pooling_nd > 0:
            
                pool_size = attn_pool_size
                head_num, _ = val_norm_score.shape
                frame_num = val_norm_score.shape[-1] // token_per_frame
                
                if avg_pooling_nd == 1:
                    avg_pool = torch.nn.AvgPool1d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
                elif avg_pooling_nd == 2:
                    avg_pool = torch.nn.AvgPool2d(kernel_size=(pool_size, pool_size), stride=(1, 1), padding=(pool_size // 2, pool_size // 2))
                elif avg_pooling_nd == 3:
                    avg_pool = torch.nn.AvgPool3d(kernel_size=(pool_size, pool_size, pool_size), stride=(1, 1, 1), padding=(pool_size // 2, pool_size // 2, pool_size // 2))
                else:
                    raise NotImplementedError("We support 1D - 3D Pooling")
                
                patch_size = model.config.height_width if hasattr(model.config, 'height_width') else 14  # fallback
                
                if avg_pooling_nd > 1:
                    original_numel = val_norm_score.numel()
                    val_norm_reshaped = val_norm_score.reshape(head_num, frame_num, patch_size, -1)
                
                    T = val_norm_reshaped.shape[-3] 
                    H = val_norm_reshaped.shape[-2]  
                    W = val_norm_reshaped.shape[-1] 
                
                    pool_size_eff = min(pool_size, T, H, W)

                    if pool_size_eff % 2 == 0:
                        pool_size_eff -= 1
                
                    if pool_size_eff < 2:
                        pass
                    else:
                        if pool_size_eff != pool_size:
                            pool_size = pool_size_eff
                            if avg_pooling_nd == 2:
                                avg_pool = torch.nn.AvgPool2d(
                                    kernel_size=(pool_size, pool_size),
                                    stride=(1, 1),
                                    padding=(pool_size // 2, pool_size // 2),
                                )
                            elif avg_pooling_nd == 3:
                                avg_pool = torch.nn.AvgPool3d(
                                    kernel_size=(pool_size, pool_size, pool_size),
                                    stride=(1, 1, 1),
                                    padding=(pool_size // 2, pool_size // 2, pool_size // 2),
                                )
                
                        val_norm_pooled = avg_pool(val_norm_reshaped)
                        val_norm_score = val_norm_pooled.reshape(head_num, -1)
                        assert original_numel == val_norm_score.numel()
                # 1D pooling
                else:
                    val_norm_score = avg_pool(val_norm_score)
            
            # Combine tar and Value norm scoring
            if tar_indices is not None:
                head_indices = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, tar_indices.size(1))
                val_norm_score[head_indices, tar_indices] = val_norm_score.max() + 1
            
            # Final selection based on combined scores
            all_indices = val_norm_score.topk(compress_frame_num * token_per_frame, dim=-1).indices
            all_indices, _ = all_indices.sort(dim=-1)
            
        else:
            raise ValueError(f"Unknown compression method: {method}. Available methods: 'swa', 'uniform', 'tar', 'val_norm', 'tar_val'")
        
        # Apply compression using advanced indexing
        batch_indices = torch.zeros_like(all_indices)
        head_indices = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, all_indices.size(1))
        
        # Ensure indices are within bounds
        assert all_indices.max() <= key_states_to_compress.shape[2], \
            f"Selected index max {all_indices.max()} exceeds key length {key_states_to_compress.shape[2]}"
        
        # Gather selected tokens
        new_key_states_to_compress = key_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        new_value_states_to_compress = value_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        
        # Concatenate with system tokens
        new_k = torch.cat([key_states[:, :, :system_size, :], new_key_states_to_compress], dim=2)
        new_v = torch.cat([value_states[:, :, :system_size, :], new_value_states_to_compress], dim=2)

        past_key_values.layers[layer_idx].keys = new_k
        past_key_values.layers[layer_idx].values = new_v

    return past_key_values, None