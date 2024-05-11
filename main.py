#!/usr/bin/env python
import torch
from transformers import AutoProcessor, AutoModelForPreTraining
from transformers import AutoTokenizer, AutoModelForCausalLM

from PIL import Image
import requests
import copy

def merge(target: torch.Tensor, master: torch.Tensor, base: torch.Tensor) -> torch.Tensor: # Add on the target model with master model and remove the overlap base model.
    return target + master - base

def main(master: str, target: str, base, save: str, tokenizer: str = None, skip: list = ['model.embed_tokens.weight', 'lm_head.weight']) -> None:
    # Load models
    print("Loading model")
    model_base = AutoModelForCausalLM.from_pretrained( base )
    model_target = AutoModelForCausalLM.from_pretrained( target )
    model_master = AutoModelForCausalLM.from_pretrained( master )
    tk = AutoTokenizer.from_pretrained(master if tokenizer == None else tokenizer)
    # Resize the embedding
    print("Align model embedding")
    model_target.resize_token_embeddings(model_master.model.embed_tokens.weight.size(0))
    
    # Operationable model
    op_master = model_master.state_dict()
    op_base   =   model_base.state_dict()

    # Process the model
    print("Start processing...")
    for param_name, param_value in model_target.named_parameters():
        if param_name in skip:
            title = f"=== {param_name} ==="
            print(title)
            print(param_value.data)
            param_value.data = op_master[param_name]
            print(param_value.data)
            print("="*len(title))
        else:
            pass
            title = f"=== {param_name} ==="
            print(title)
            print("Original:", param_value.data)
            param_value.data = merge(param_value, op_master[param_name], op_base[param_name])
            #param_value.data += op_master[param_name] - op_base[param_name]
            print("New:", param_value.data)
            print("="*len(title))
    
    print("++++++ Model process successfully ++++++")
    
    # Need to check the saving path "save"
    print("Saving model...")
    model_target.save_pretrained(save, from_pt=True)
    tk.save_pretrained(save, from_pt=True)



if __name__ == "__main__":
    import fire
    fire.Fire(main)
