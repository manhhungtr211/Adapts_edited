from transformers import AutoModelForCausalLM
import torch
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def get_model(model=None, **kwargs):
    """Load hoặc trả về model cho inference."""
    
    model_parallel = kwargs.pop("model_parallel", False)
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    model_name = kwargs.get("pretrained_model_name_or_path", "")
    use_parallel = str(model_parallel).lower() == "true"

    if model is not None:
        print("[get_model] Sử dụng model object được truyền trực tiếp")
        return model

    # ========================================================
    # XỬ LÝ RIÊNG: Bypass HuggingFace API cho mô hình TRM
    # ========================================================
    if "trm-convfinqa" in model_name:
        print(f"\n[get_model] Kích hoạt chế độ load THỦ CÔNG cho {model_name}...")
        from src.models.trm_model import TinyRecursiveModel, FinanceTRMWrapper
        
        # 1. Tự động tải file safetensors từ mạng về máy
        print("-> Đang tải file weights (.safetensors) từ HuggingFace Hub...")
        weight_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        
        # 2. Khởi tạo cấu trúc mạng
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
        trm_model = TinyRecursiveModel(**config)
        
        # 3. Nạp weights vào mạng
        print("-> Đang nạp weights vào mô hình...")
        state_dict = load_file(weight_path)
        # Dùng strict=False để bỏ qua các sai lệch nhỏ về tên key nếu có
        trm_model.load_state_dict(state_dict, strict=False)
        
        # 4. Đẩy lên GPU (nếu có)
        if use_parallel:
            trm_model = trm_model.cuda()
            
        model = FinanceTRMWrapper(trm_model)
        
        if use_parallel and torch.cuda.device_count() > 1:
            print("-> Wrapping TRM with DataParallel...")
            model.model = torch.nn.DataParallel(model.model)
            
        print("-> Load mô hình thành công!\n")
        return model
    # ========================================================

    # DÀNH CHO CÁC MÔ HÌNH CHUẨN KHÁC (ví dụ: Llama, Mistral...)
    if use_parallel:
        n_gpus = torch.cuda.device_count()
        max_memory = {i: "8GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "30GiB" 

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
"""
from transformers import AutoModelForCausalLM
import torch
import os

from transformers.dynamic_module_utils import get_class_from_dynamic_module
original_get_class = get_class_from_dynamic_module

def patched_get_class(*args, **kwargs):
    cls = original_get_class(*args, **kwargs)
    if not hasattr(cls, "register_for_auto_class"):
        cls.register_for_auto_class = classmethod(lambda c, *a, **k: None)
    return cls

import transformers.dynamic_module_utils
transformers.dynamic_module_utils.get_class_from_dynamic_module = patched_get_class

def get_model(model=None, **kwargs):
    
    Load hoặc trả về model cho inference.

    Args:
        model: (Optional) Model object đã được load/fine-tune sẵn.
               Nếu được truyền vào, bỏ qua việc load từ pretrained_model_name_or_path.
        **kwargs: Các tham số khác (pretrained_model_name_or_path, cache_dir,
                  trust_remote_code, model_parallel, checkpoint_path, ...).

    Returns:
        Model sẵn sàng cho inference.
    
    model_parallel = kwargs.pop("model_parallel", False)
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    # Fix: "True" (string) != True (bool)
    use_parallel = str(model_parallel).lower() == "true"

    # Nếu đã có model object (vừa fine-tune hoặc load thủ công) → trả luôn
    if model is not None:
        print("[get_model] Sử dụng model object được truyền trực tiếp (đã fine-tune hoặc load sẵn)")
        return model

    # Load TRM model from .pt checkpoint
    if checkpoint_path is not None and str(checkpoint_path) != "null":
        print(f"[get_model] Loading TRM from checkpoint: {checkpoint_path}")
        from src.models.trm_model import TinyRecursiveModel, FinanceTRMWrapper
        
        # Hardcoded config matching the notebook (updated for Llama-2 tokenizer)
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
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if 'model_state_dict' in ckpt:
            trm_model.load_state_dict(ckpt['model_state_dict'])
        else:
            trm_model.load_state_dict(ckpt)
            
        # Wrap it to mimic HuggingFace model
        if use_parallel:
            trm_model = trm_model.cuda()
        model = FinanceTRMWrapper(trm_model)
        
        if use_parallel and torch.cuda.device_count() > 1:
            print("[get_model] Wrapping TRM with DataParallel...")
            model.model = torch.nn.DataParallel(model.model)
            
        return model

    if use_parallel:
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
"""