from transformers import AutoModelForCausalLM
import torch

def get_model(**kwargs):
    model_parallel = kwargs.pop("model_parallel")
    # Fix: "True" (string) != True (bool)
    use_parallel = str(model_parallel).lower() == "true"

    if use_parallel:
        import torch
        n_gpus = torch.cuda.device_count()
        # Để lại ~2GB headroom mỗi GPU cho activations/KV cache
        max_memory = {i: "14GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "30GiB"  # fallback nếu model quá lớn

        model = AutoModelForCausalLM.from_pretrained(
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            **kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            torch_dtype=torch.float16,
            **kwargs
        )
    return model