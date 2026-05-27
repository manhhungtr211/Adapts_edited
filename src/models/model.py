from transformers import AutoModelForCausalLM
import torch
import os

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
    
    model_name_or_path = kwargs.get("pretrained_model_name_or_path", "")

    # Nếu đã có model object (vừa fine-tune hoặc load thủ công) → trả luôn
    if model is not None:
        print("[get_model] Sử dụng model object được truyền trực tiếp (đã fine-tune hoặc load sẵn)")
        return model

    # Check if the path points to a TRM model (.pt file)
    if isinstance(model_name_or_path, str) and model_name_or_path.endswith('.pt'):
        print(f"[get_model] Detected TRM model file: {model_name_or_path}. Loading custom TRM...")
        from src.models.trm_model import TinyRecursiveModel, FinanceTRMWrapper
        
        # Hardcoded config matching the notebook
        config = {
            'vocab_size': 50257,
            'dim': 256,
            'n_heads': 8,
            'n_layers': 4,
            'mlp_ratio': 16,
            'max_seq_len': 128,
            'n_latent_recursions': 4,
            'n_improvement_cycles': 2,
        }
        
        trm_model = TinyRecursiveModel(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            mlp_ratio=config['mlp_ratio'],
            max_seq_len=config['max_seq_len'],
            n_latent_recursions=config['n_latent_recursions'],
            n_improvement_cycles=config['n_improvement_cycles'],
        )
        
        checkpoint = torch.load(model_name_or_path, map_location="cpu", weights_only=True)
        if 'model_state_dict' in checkpoint:
            trm_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            trm_model.load_state_dict(checkpoint)
            
        # Wrap it to mimic HuggingFace model
        model = FinanceTRMWrapper(trm_model)
        
        if use_parallel and torch.cuda.device_count() > 1:
            print("[get_model] Wrapping TRM with DataParallel...")
            model.model = torch.nn.DataParallel(model.model)
            
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