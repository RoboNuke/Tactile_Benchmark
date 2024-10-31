# logging data
wandb_entity="hur"
wandb_project_name="In-Contact_Baseline"
#wandb_project_name="Tester"
save_model=1
capture_video=1

# create folder for this round of experiments
gpu_path="/nfs/stak/users/brownhun/hpc-share/Tactile_Benchmark/learning/HPC/hpc_ppo.py"


# shared learning data
#env_id="PegInsertionSide-v1"
#env_id="FragilePegInsert-v1"
if [ "$4" = 'rgb' ] || [ "$4" = 'rgb_no_ft' ]; then
    num_envs=128
    num_minibatches=75 # ensures minibatch size of 256
else
    num_envs=256
    num_minibatches=150 # ensures minibatch size of 256
fi

eval_freq=10
update_epochs=2
partial_reset=1
reconfiguration_freq=1
reward_scale=1.0

# exp data
start=$1
end=$2
env_id=$3
obs_mode=$4
dmg_force=$5
exp_set_name=$6 #"Stability_Baseline"
num_steps=$7
total_timesteps=$8
control_mode=$9
reward_mode=${10}
force_encoding=${11}
critic_n=${12}
critic_l=${13}
use_shampoo=${14}
lock_gripper=${15}
ent_co=${16}
lr=${17}

if [[ $obs_mode == *"no_ft"* ]]; then
    include_force=0
else
    include_force=1
fi

if [[ $obs_mode == *"rgb"* ]]; then
    include_state=0
else
    include_state=1
fi

date=$(date +"%Y-%m-%d_%H:%M")
#_${date}"
if [ $lock_gripper -eq 1 ]; then
    lock_gripper='lock_gripper'
else
    lock_gripper='no-lock-gripper'
fi

if [ $save_model -eq 1 ]; then
    save_model='save_model'
else
    save_model='no-save-model'
fi

if [ $capture_video -eq 1 ]; then
    capture_video='capture-video'
else
    capture_video='no-capture-video'
fi

if [ $partial_reset -eq 1 ]; then
    partial_reset='partial-reset'
else
    partial_reset='no-partial-reset'
fi

if [ $include_force -eq 1 ]; then
    include_force='include-force'
else
    include_force='no-include-force'
fi

if [ $include_state -eq 1 ]; then
    include_state='include-state'
else
    include_state='no-include-state'
fi

if [ $use_shampoo -eq 1 ]; then
    use_shampoo='use-shampoo'
else
    use_shampoo='no-use-shampoo'
fi

for i in $(seq $start $end);
do
    #printf "\n\n\n\nStarting baseline exp ${i}\n\n\n\n"
    #exp_name = "pickcube_state_baseline_" + $i
    exp_name="${exp_set_name}_${i}" #_${date}"
    python $gpu_path \
        --wandb-project-name=$wandb_project_name \
        --wandb-entity=$wandb_entity \
        --$save_model \
        --$capture_video \
        --env-id=$env_id \
        --num-envs=$num_envs \
        --num-steps=$num_steps \
        --num_eval_steps=$num_steps \
        --total-timesteps=$total_timesteps \
        --eval-freq=$eval_freq \
        --update-epochs=$update_epochs \
        --num-minibatches=$num_minibatches \
        --$partial_reset \
        --reconfiguration-freq=$reconfiguration_freq \
        --reward-scale=$reward_scale \
        --obs-mode=$obs_mode \
        --control-mode=$control_mode \
        --reward-mode=$reward_mode \
        --force-encoding=$force_encoding \
        --$include_force \
        --$include_state \
        --seed=$i \
        --exp-name=$exp_name \
        --exp-max-dmg-force=$dmg_force \
        --critic_n=$critic_n \
        --critic_hidden_layer_size=$critic_l \
        --$use_shampoo \
        --$lock_gripper \
        --ent-coef=$ent_co \
        --learning-rate=$lr
done



# store data
#python -m data_analysis.tb_extraction $filepath state $number_of_seeds "runs/${filepath}/"

# store plots
#mkdir -p "runs/${filepath}/plots/"
#python -m data_analysis.pd_graphing "runs/${filepath}/"
