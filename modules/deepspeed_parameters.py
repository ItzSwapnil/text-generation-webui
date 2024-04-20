def generate_offload_config(nvme_offload_dir, train_batch_size):
    if nvme_offload_dir:
        offload_device = "nvme"
        offload_path = nvme_offload_dir
        buffer_count = 5
        buffer_size = 1e9
    else:
        offload_device = "cpu"
        offload_path = None
        buffer_count = None
        buffer_size = None

    return {
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": offload_device,
                "nvme_path": offload_path,
                "pin_memory": True,
                "buffer_count": buffer_count,
                "buffer_size": buffer_size,
                "max_in_cpu": 1e9 if offload_device == "cpu" else None,
            },
            "overlap_comm": True,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
            "sub_group_size": 1e8,
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": "auto",
            "stage3_max_reuse_distance": "auto",
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True,
        },
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }


def generate_ds_config(ds_bf16, train_batch_size, nvme_offload_dir):
    """
    DeepSpeed configuration
    https://huggingface.co/docs/transformers/main_classes/deepspeed
    """

    ds_config = {
        "fp16": {"enabled": not ds_bf16},
        "bf16": {"enabled": ds_bf16},
    }

    ds_config.update(generate_offload_config(nvme_offload_dir, train_batch_size))

    return ds_config
