# Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence

A joint project by: Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, Micah Goldblum

<p align="center">
<a target="_blank" href="https://arxiv.org/abs/2511.07384">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-B31B1B?style=flat&logo=arxiv">
<a target="_blank" href="https://hf.co/collections/tomg-group-umd/retrofitting-recurrence">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<br>
</p>


## Citing Our Work
To cite our work, please use this bibtex.
```
@article{mcleish2025teaching,
    title={Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence}, 
    author={Sean McLeish and Ang Li and John Kirchenbauer and Dayal Singh Kalra and Brian R. Bartoldson and Bhavya Kailkhura and Avi Schwarzschild and Jonas Geiping and Tom Goldstein and Micah Goldblum},
    journal={arXiv preprint arXiv:2511.07384},
    year={2025}
}
```

# Reproducing Experiments
## Getting Started
We developed in Python 3.11, to install run:
```
git clone git@github.com:mcleish7/retrofitting-recurrence.git
cd retrofitting-recurrence
pip install -r requirements.txt
```

## Datasets
1. To download run: `python utils/download_ds.py --dataset_path YOUR_DATASET_LOCATION` Note: you may need to request permission on HuggingFace to access some of the Nvidia datasets.
2. To tokenize run: `python preprocess_data_packing.py --out_path="llama_1b_packed_nemotron_cc_math_v1_4plus_wrapped_packing" --dataset_location="datasets/Nemotron-CC-Math-v1-4plus" --cache_path=YOUR_CACHE_PATH --save_path=YOUR_SAVE_PATH` You can use the `tokenizer_name` flag to control the tokenizer being used.
3. To save to parquet: `python utils/to_parquet.py --dataset_path YOUR_TOKENIZED_DATASET_LOCATION --dataset_save_dir $PROCESSED_DATA_PATH/YOUR_PARQUET_SAVE_LOCATION`

As an example, we upload our Llama-3 tokenized parquet FineWeb-Edu-350B dataset [here](https://huggingface.co/datasets/smcleish/retrofitting-llama-fineweb-edu-tokenized).

### Mixing Datasets
To obtain the data mix used in Figure 8, we run [mix_datasets.py](mix_datasets.py). We split our datasets into shards to process and tokenize them and combine some shards in `mix_datasets.py`, if your workflow is different, we take approximately 12.8M rows from each split. We note due to the Nemotron licence, we cannot openly upload our exact dataset; please open an issue if there is any trouble here. 

## Converting a Pretrained Model
We provide conversion scripts for TinyLlama, Llama and OLMo in [convert_pretrained_model](convert_pretrained_model) and provide untrained (outputs from the conversion script) models in our [collection](https://huggingface.co/collections/tomg-group-umd/retrofitting-recurrence).
- For TinyLlama/Llama use [convert_pretrained_model/convert_llama.py](convert_pretrained_model/convert_llama.py)
- For OLMo use [convert_pretrained_model/convert_olmo.py](convert_pretrained_model/convert_olmo.py)

There are multiple steps, we are going to use multiple files to ensure that the converted model is as faithful to the original as possible:
1. Download the model using [utils/download_to_local.py](utils/download_to_local.py). Also, download `tomg-group-umd/huginn-0125` at `revision="972cea674c2f4ea37da6777ece1a0c9895c9998b"` into `convert_pretrained_model/models/huginn-0125`.
2. Add the `looped_{model}.py` file into the downloaded snapshot dir. 
3. Run `convert_{model}.py` code (read the comment at the top of the main function for how to select different model shapes), this will error but the dir with the new weights will be created.
4. In the newly created dir, overwrite the contents of `raven_modeling_minimal.py` file with the contents of `raven_modeling_minimal_compare_{model}.py` file.
5. Rerun `convert_{model}.py`, this time you should see a lot of `True, 0.000` printed meaning that the hidden states all match exactly. If not there is something wrong, reread all variables changed in `convert_{model}.py`, fix and retry.
6. Overwrite the contents of `raven_modeling_minimal.py` file with the contents of `raven_modeling_minimal_{model}.py` file. This is slightly different to compare in that it returns less information and uses the linear adapter.

NOTE: the model conversion code is built to work with `transformers==4.51.0` due to a KV-Cache breaking change in future versions.

WARNING: We only tested the parts of the modelling files used in this repo (e.g. forward(), generate()), however leave in all functions from the `Huginn-0125` model. Please use untested features with caution.

## Training
Example commands are in the [shells/](shells/) directory, organised by model. We use the same `$PROCESSED_DATA_PATH` temporary variable as used in the datasets section above, make sure to overwrite this to your specific path.

We use the [train.py](train.py) to train, this is based on the [Huginn finetuning script](https://github.com/seal-rg/recurrent-pretraining/blob/main/finetuning_simple_example.py) but with extra features, such as parquet data loading and extra optimizers.
Note the `save_n_mins_before_timeout` flag is designed to work on [flux scheduling](https://computing.llnl.gov/projects/flux-building-framework-resource-management) systems only.

## Evals
Example commands are in the [shells/eval.sh](shells/eval.sh) file using [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness).
We added " Let's think step by step." to the gsm8k-cot prompt, our yaml is in [eval_yamls/gsm8k-cot-sean.yaml](eval_yamls/gsm8k-cot-sean.yaml), place this alongside the `gsm8k.yaml` in lm_eval.

For offline validation loss calculations (as there is no training val loop), use [multi_recurence_eval.py](multi_recurence_eval.py). Example command in bottom of python file.

NOTE: If you get an error like: "TypeError: ... got multiple values for keyword argument 'tie_word_embeddings'", remove the `tie_word_embeddings` key from the models config.json, as Huginn-0125 uses the `tie_embeddings` flag instead.

## Analysis
We provide plotting code in [plot_evals.py](plot_evals.py), which is useful for plotting multiple experiments at once quickly. I have left an example of how I would plot my olmo runs here.

We provide the exact plotting code and data used in our paper in [paper_plots/](paper_plots/). Run [paper_plots/plot.py](paper_plots/plot.py) to recreate the plots.

## Misc
1. To untie the embeddings and lm_head for non-recurrent Llama models before training use [utils/untie_embeds_hf.py](utils/untie_embeds_hf.py).
2. For ShortGPT experiments we use https://github.com/sramshetty/ShortGPT.

# Contact
Please, feel free to contact us with any questions, or open an issue on Github.
