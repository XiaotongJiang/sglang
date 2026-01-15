def prepare_static_weights_for_trtllm_fp4_moe(
    gemm1_weights,
    gemm2_weights,
    gemm1_scales_linear_fp4_bytes,
    gemm2_scales_linear_fp4_bytes,
    hidden_size,
    intermediate_size,
    num_experts,
):
    """Prepare quantized weights for kernel (done offline with weights).
    
    Uses shuffle_matrix_a and shuffle_matrix_sf_a which handle non-128-aligned dimensions,
    same approach as MXFP4 weights.
    """
    from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a
    
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

    # Convert quantized weights to proper formats
    gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )  # packed fp4
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 16
    )  # fp8 scaling factors

    gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )  # packed fp4
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, hidden_size, intermediate_size // 16
    )  # fp8 scaling factors

    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    
    for i in range(num_experts):
        # Use shuffle_matrix_a which handles non-128-aligned dimensions
        gemm1_weights_fp4_shuffled.append(
            shuffle_matrix_a(gemm1_weights_fp4[i].view(torch.uint8), epilogue_tile_m)
        )
        gemm1_scales_fp4_shuffled.append(
            shuffle_matrix_sf_a(
                gemm1_scales_linear_fp4[i].view(torch.uint8), epilogue_tile_m
            )
        )
        gemm2_weights_fp4_shuffled.append(
            shuffle_matrix_a(gemm2_weights_fp4[i].view(torch.uint8), epilogue_tile_m)
        )
        gemm2_scales_fp4_shuffled.append(
            shuffle_matrix_sf_a(
                gemm2_scales_linear_fp4[i].view(torch.uint8), epilogue_tile_m
            )
        )

    # Stack weights for all experts
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = (
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
    )

    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = (
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    
    return (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
        hidden_size,  # Return original hidden_size (no padding needed)
        intermediate_size,  # Return original intermediate_size (no padding needed)
    )

