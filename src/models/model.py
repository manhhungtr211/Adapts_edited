from transformers import AutoModelForCausalLM
import torch

def get_model(**kwargs):
    if kwargs.pop("model_parallel")==True:
        model = AutoModelForCausalLM.from_pretrained(device_map="auto", torch_dtype=torch.float16, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(torch_dtype=torch.float16, **kwargs)
    return model