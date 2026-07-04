# To run any of these evals for non_recurrent models simply remove `mean_recurrence=${MEAN_RECURRENCE},`
OUT_ROOT="eval_outputs"
MODEL_PATH=
chkpt=
MEAN_RECURRENCE=

HIP_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH}/model_only_chkpt_${chkpt},mean_recurrence=${MEAN_RECURRENCE},add_bos_token=True,dtype="float32",trust_remote_code=True \
    --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,mmlu,openbookqa,piqa,social_iqa,winogrande,asdiv \
    --device cuda \
    --output_path "${OUT_ROOT}/${MODEL_PATH}/model_only_chkpt_${chkpt}" \
    --batch_size auto &

HIP_VISIBLE_DEVICES=1 lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH}/model_only_chkpt_${chkpt},mean_recurrence=${MEAN_RECURRENCE},add_bos_token=True,dtype="float32",trust_remote_code=True,max_length=1024 \
    --tasks gsm8k_cot_sean,minerva_math \
    --device cuda \
    --output_path "${OUT_ROOT}/${MODEL_PATH}/model_only_chkpt_${chkpt}/model_only_chkpt_${chkpt}_gsm8k_cot_sean_minerva_math.json" \
    --batch_size 32 --num_fewshot=1 &