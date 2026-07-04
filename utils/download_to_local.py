from huggingface_hub import snapshot_download

local_path = snapshot_download(repo_id="meta-llama/Llama-3.2-1B", local_dir="models/Llama-3.2-1B")
# local_path = snapshot_download(repo_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", local_dir="convert_pretrained_model/models/TinyLlama-1.1B-intermediate-step-1431k-3T")
# local_path = snapshot_download(
#     repo_id="allenai/OLMo-2-0425-1B",
#     revision="stage1-step1907359-tokens4001B",
#     local_dir="convert_pretrained_model/models/OLMo-2-0425-1B-step1907359",
# )
# local_path = snapshot_download(
#     repo_id="tomg-group-umd/huginn-0125",
#     revision="972cea674c2f4ea37da6777ece1a0c9895c9998b",
#     local_dir="convert_pretrained_model/models/huginn-0125"
# )

print(local_path)