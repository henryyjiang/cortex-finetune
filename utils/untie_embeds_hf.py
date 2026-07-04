from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch

model_name = "models/Llama-3.2-1B"
cfg = AutoConfig.from_pretrained(model_name)
cfg.tie_word_embeddings = False
model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg)

with torch.no_grad():
    inp = model.get_input_embeddings() # nn.Embedding
    out = model.get_output_embeddings() # nn.Linear
    out.weight.copy_(inp.weight)

print("same after copy? ", torch.equal(model.get_input_embeddings().weight, model.get_output_embeddings().weight))

# print(model.get_output_embeddings())
if getattr(model, "_keys_to_ignore_on_save", None):
    print(model._keys_to_ignore_on_save)
    model._keys_to_ignore_on_save = [
        k for k in model._keys_to_ignore_on_save
        if "lm_head.weight" not in k
    ]

model.save_pretrained(f"{model_name}-untied")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(f"{model_name}-untied")

model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg)
untied_model, info = AutoModelForCausalLM.from_pretrained(f"{model_name}-untied", output_loading_info=True)

if torch.equal(model.get_input_embeddings().weight,model.get_output_embeddings().weight):
    print("BAD: og model weights same")
else:
    print("GOOD: og model weights diff")

print(untied_model.get_input_embeddings().weight)
print(untied_model.get_output_embeddings().weight)
if torch.equal(untied_model.get_input_embeddings().weight, untied_model.get_output_embeddings().weight):
    print("GOOD: new model weights same")
else:
    print("BAD: new model weights diff")

if untied_model.get_input_embeddings().weight.data_ptr() == untied_model.get_output_embeddings().weight.data_ptr():
    print("BAD: new model pointers same")
else:
    print("GOOD: new model pointers diff")

if torch.equal(model.get_input_embeddings().weight, untied_model.get_output_embeddings().weight):
    print("GOOD: new model lm_head matched old model embeds same")
else:
    print("BAD: new model lm_head diff old model embeds")

if torch.equal(untied_model.get_input_embeddings().weight, model.get_output_embeddings().weight):
    print("BAD: new model embeds match old model lm_head same")
else:
    print("GOOD: new model embeds diff old model lm_head")