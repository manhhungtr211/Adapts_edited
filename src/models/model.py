from transformers import AutoModelForCausalLM
import torch

def get_model(**kwargs):
    model_parallel = kwargs.pop("model_parallel")
    # Fix: "True" (string) != True (bool)
    use_parallel = str(model_parallel).lower() == "true"
    
    if use_parallel:
        model = AutoModelForCausalLM.from_pretrained(
            device_map="auto",
            torch_dtype=torch.float16,
            **kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            torch_dtype=torch.float16,
            **kwargs
        )
    return model