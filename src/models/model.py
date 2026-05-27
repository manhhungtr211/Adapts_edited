from transformers import AutoModelForCausalLM
import torch

def get_model(model=None, **kwargs):
    """Load hoặc trả về model cho inference.

    Args:
        model: (Optional) Model object đã được load/fine-tune sẵn.
               Nếu được truyền vào, bỏ qua việc load từ pretrained_model_name_or_path.
        **kwargs: Các tham số khác (pretrained_model_name_or_path, cache_dir,
                  trust_remote_code, model_parallel, ...).

    Returns:
        Model sẵn sàng cho inference.
    """
    model_parallel = kwargs.pop("model_parallel", False)
    # Fix: "True" (string) != True (bool)
    use_parallel = str(model_parallel).lower() == "true"

    # Nếu đã có model object (vừa fine-tune hoặc load thủ công) → trả luôn
    if model is not None:
        print("[get_model] Sử dụng model object được truyền trực tiếp (đã fine-tune hoặc load sẵn)")
        return model

    if use_parallel:
        import torch
        n_gpus = torch.cuda.device_count()
        # Giới hạn 8GiB/GPU để device_map chia đều layer,
        # nhường ~6GB headroom cho activations + KV cache trong lúc inference
        max_memory = {i: "8GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "30GiB"  # overflow nếu model > tổng GPU memory

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