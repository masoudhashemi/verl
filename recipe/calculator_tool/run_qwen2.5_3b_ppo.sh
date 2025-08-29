set -x

# ================= paths =================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

train_files="$DATA_ROOT/data/calculator_tool/train.parquet"
test_files="$DATA_ROOT/data/calculator_tool/test.parquet"

# model
actor_model_path=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
critic_model_path=${CRITIC_MODEL_PATH:-$actor_model_path}

# tool
tool_config_path=recipe/calculator_tool/calculator_tool_config.yaml

# wandb / logging
project_name=calculator_tool_aspo
experiment_name=qwen2.5_3b_ppo
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
adv_estimator=gae
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=6
max_prompt_length=1024
max_response_length=1024
actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=1.0

critic_warmup=10

train_batch_size=16
ppo_mini_batch_size=8
n_resp_per_prompt_val=2

# ================= performance =================
infer_tp=1
train_sp=1

offload=True

actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 2))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    algorithm.aspo.enabled=true \
    algorithm.aspo.delta=-2.0 \
    algorithm.aspo.k=0.7 \
    algorithm.aspo.require_code_pass=false \
    algorithm.aspo.code_detection.prefer_structured_tool_calls=true \
    algorithm.aspo.code_detection.markdown_backticks=true \
    algorithm.aspo.code_detection.html_code_tags=true \
    data.train_files="['$train_files']" \
    data.val_files="['$test_files']" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path=recipe/calculator_tool/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    critic.optim.lr=$critic_lr \
    critic.model.use_remove_padding=True \
    critic.model.path=$critic_model_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.ulysses_sequence_parallel_size=$train_sp \
    critic.model.fsdp_config.param_offload=$offload \
    critic.model.fsdp_config.optimizer_offload=$offload \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=1 \
    trainer.total_epochs=1 $@

