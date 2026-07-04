import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os

def get_edited_model(model_name, extra_args={}):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if ("llama" in model_name.lower()):
        config_args = {
            "model_type": "looped_llama2",
            "auto_map": {"AutoModelForCausalLM": "looped_llama.LoopedLlamaForCausalLM"},
            "architectures": ["LoopedLlamaForCausalLM"],
        }
    elif ("olmo-2" in model_name.lower()):
        config_args = {
            "model_type": "looped_olmo2",
            "auto_map": {"AutoModelForCausalLM": "looped_olmo.LoopedOlmo2ForCausalLM"},
            "architectures": ["LoopedOlmo2ForCausalLM"],
        }
    else:
        print("model not found")
        exit()

    config.__dict__.update(config_args)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        attn_implementation="sdpa",
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    model.rec_post_init(extra_args, {})
    return model


def force_attn_impl(name):
    if name == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif name == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        print("attn impl not found")
        exit()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def get_llama_huginn_config(llama_config_name):
    config = AutoConfig.from_pretrained("models/huginn-0125-with-qk-norm-olmo2", trust_remote_code=True)
    llama_config = AutoConfig.from_pretrained(llama_config_name, trust_remote_code=True)
    # print(config)
    if llama_config.tie_word_embeddings:
        print("llama model has tied embeddings but this models won't have (\"tie_embeddings\": False), check you mean this")
        # exit()
    update_dict = {
        "head_dim": int(llama_config.hidden_size / llama_config.num_attention_heads),
        "intermediate_size": llama_config.intermediate_size, 
        "n_embd": llama_config.hidden_size,
        "n_heads": llama_config.num_attention_heads,
        "num_key_value_heads": llama_config.num_key_value_heads, #8,#32,
        "n_layers": 14,
        "n_layers_in_coda": 4,
        "n_layers_in_prelude": 4,
        "n_layers_in_recurrent_block": 6,
        "norm_eps": llama_config.rms_norm_eps,
        "vocab_size": llama_config.vocab_size,
        "padded_vocab_size": llama_config.vocab_size,
        "rope_base": 500000.0,
        "tie_embeddings": False,
        "torch_dtype": llama_config.torch_dtype,
        "qk_bias": False,
        "max_position_embeddings": llama_config.max_position_embeddings
    }


    for key, value in update_dict.items():
        setattr(config, key, value)

    config.init_values["embed_scale"] = 1.0
    if llama_config.rope_theta:
        config.rope_theta = llama_config.rope_theta

    # print(config)
    return config

def get_looped_llama(model_name, looped_args):
    model = get_edited_model(model_name, looped_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def weight_mapping(llama_state_dict, huginn_state_dict, mapping_cfg):
    # 0. transfer token embeddings & lm head (shape-compatible)
    huginn_state_dict["transformer.wte.weight"] = llama_state_dict["model.embed_tokens.weight"]
    huginn_state_dict["lm_head.weight"] = llama_state_dict["lm_head.weight"]
    huginn_state_dict["transformer.ln_f.weight"] = llama_state_dict["model.norm.weight"]

    def copy_layer(src_i, tgt_prefix):
        """
        helper to copy a single layer
        """
        # attn
        q_w = llama_state_dict[f"model.layers.{src_i}.self_attn.q_proj.weight"]
        k_w = llama_state_dict[f"model.layers.{src_i}.self_attn.k_proj.weight"]
        v_w = llama_state_dict[f"model.layers.{src_i}.self_attn.v_proj.weight"]

        # cat along out-features → (n_embd + 2*n_kv*hdim, n_embd)
        huginn_state_dict[f"{tgt_prefix}.attn.Wqkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        huginn_state_dict[f"{tgt_prefix}.attn.proj.weight"] = llama_state_dict[f"model.layers.{src_i}.self_attn.o_proj.weight"]

        # MLP
        gate_proj = llama_state_dict[f"model.layers.{src_i}.mlp.gate_proj.weight"]
        up_proj = llama_state_dict[f"model.layers.{src_i}.mlp.up_proj.weight"]
        huginn_state_dict[f"{tgt_prefix}.mlp.fc.weight"] = torch.cat([gate_proj, up_proj], dim=0)
        huginn_state_dict[f"{tgt_prefix}.mlp.proj.weight"] = llama_state_dict[f"model.layers.{src_i}.mlp.down_proj.weight"]

        if f"model.layers.{src_i}.self_attn.q_norm.weight" in llama_state_dict:
            huginn_state_dict[f"{tgt_prefix}.attn.q_norm.weight"] = llama_state_dict[f"model.layers.{src_i}.self_attn.q_norm.weight"]
            huginn_state_dict[f"{tgt_prefix}.attn.k_norm.weight"] = llama_state_dict[f"model.layers.{src_i}.self_attn.k_norm.weight"]

        # LayerNorms
        huginn_state_dict[f"{tgt_prefix}.norm_1.weight"] = llama_state_dict[f"model.layers.{src_i}.post_attention_layernorm.weight"]
        huginn_state_dict[f"{tgt_prefix}.norm_2.weight"] = llama_state_dict[f"model.layers.{src_i}.post_feedforward_layernorm.weight"]
        
    # 2. prelude → core → coda
    for j, src_i in enumerate(mapping_cfg["prelude_idx"]):
        copy_layer(src_i, f"transformer.prelude.{j}")

    for j, src_i in enumerate(mapping_cfg["core_idx"]):
        copy_layer(src_i, f"transformer.core_block.{j}")

    for j, src_i in enumerate(mapping_cfg["coda_idx"]):
        copy_layer(src_i, f"transformer.coda.{j}")

    return huginn_state_dict


def get_llama_huginn(looped_llama_model, config_model_name, save_name, mapping_cfg):
    if save_name is not None:
        if os.path.exists(save_name):
            return AutoModelForCausalLM.from_pretrained(save_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    config = get_llama_huginn_config(config_model_name)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    huginn_state_dict = weight_mapping(llama_state_dict=looped_llama_model.state_dict(), huginn_state_dict=model.state_dict(), mapping_cfg=mapping_cfg)
    model.load_state_dict(huginn_state_dict)
    if save_name is not None:
        model.save_pretrained(save_name)
    return model

def check_same(looped_llama, llama_huginn, llama_tokenizer):
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = llama_tokenizer(input_text, return_tensors="pt").to(llama_huginn.device)
    looped_inputs = {k: v.clone() for k,v in inputs.items()}
    huginn_inputs = {k: v.clone() for k,v in inputs.items()}

    with torch.no_grad():
    # try:
        llama_out = looped_llama(**looped_inputs, output_hidden_states=True)
        logits_looped = llama_out.logits
    # except:
        huginn_out = llama_huginn(**huginn_inputs, output_details={"return_logits": True, "return_latents": True, "return_head": True, "return_stats": False}, num_steps=1)
        logits_huginn = huginn_out.logits

    # Compare logits
    same_shape = logits_looped.shape == logits_huginn.shape
    print(f"looped llama dtype: {looped_llama.dtype}")
    print(f"logits looped: {logits_looped.dtype}")
    print(f"llama hidden states {len(llama_out.hidden_states)}")
    print(f"llama hidden states {llama_out.hidden_states[0].shape}")
    print(f"huginn llama dtype: {llama_huginn.dtype}")
    print(f"huginn logits {logits_huginn.dtype}")
    print(f"huginn hidden states {len(huginn_out.hidden_states)}")
    print(f"huginn hidden states {huginn_out.hidden_states[0].shape}")
    print(f"huginn hidden states {len(huginn_out.latent_states)}")
    print(f"huginn hidden states {huginn_out.latent_states.shape}")
    close_values = torch.allclose(logits_looped, logits_huginn, atol=1e-4, rtol=1e-4)
    mse = torch.nn.functional.mse_loss(logits_looped, logits_huginn).item()

    print(f"Same shape: {same_shape}")
    print(f"Values close: {close_values}")
    print(f"Mean Squared Error: {mse:.6f}")

    for idx, (hug_layer, llama_layer) in enumerate(zip(huginn_out.hidden_states, llama_out.hidden_states)):
        close_values = torch.allclose(hug_layer, llama_layer, atol=1e-4, rtol=1e-4)
        mse = torch.nn.functional.mse_loss(hug_layer, llama_layer).item()
        print(f"{idx}: {close_values}, {mse:.3f}")

def main():
    """
    Places to edit:
    1. Make a copy of `models/huginn-0125` called `models/huginn-0125-with-qk-norm-olmo2` and replace the modelling file contents with the contents of `convert_pretrained_model/raven_modeling_minimal_with_qk_norm.py`
    2. `llama_model_name` and `save_name` to be the path to the feedforward model and the save path
    3. looped_args is passed to the feedforward model
    4. mapping_cfg is passed to take layers from the feedforward model to form the Raven model
    5. edit `update_dict` in `get_llama_huginn_config` to match the config input into looped_args and mapping_cfg
    """

    force_attn_impl("math")

    llama_model_name="models/OLMo-2-0425-1B-step1907359"
    save_name = "models/recurrent_olmo_2_0425_1b_step1907359_4_6_4"
    looped_args = {
        "prelude_size": 4,
        "start_index": 6,
        "block_size": 6,
        "coda_size": 4,
        "num_rec": 1,
    }
    mapping_cfg = {
        "prelude_idx": [0, 1, 2, 3],
        "core_idx": [6, 7, 8, 9, 10, 11],
        "coda_idx": [12, 13, 14, 15], 
    }

    looped_llama_model, llama_tokenizer = get_looped_llama(llama_model_name, looped_args)

    llama_huginn = get_llama_huginn(looped_llama_model, llama_model_name, save_name, mapping_cfg)
    total_params = sum(p.numel() for p in llama_huginn.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    looped_llama_model.eval().to(device=device, dtype=torch.float32)
    llama_huginn.eval().to(device=device, dtype=torch.float32)


    check_same(looped_llama_model, llama_huginn, llama_tokenizer)


if __name__ == "__main__":
    main()
