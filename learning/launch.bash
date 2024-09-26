# logging data
wandb_entity="hur"
wandb_project_name="Tester"
save_model=1
capture_video=1

# create folder for this round of experiments
exp_set_name="Stability_Baseline"
start=$1
end=$2

echo $start
echo $end

sleep 120
# shared learning data
#env_id="PegInsertionSide-v1"
env_id="FragilePegInsert-v1"
#env_id="WipeFood-v1"
num_envs=128
num_steps=150
total_timesteps=5000000
eval_freq=10

update_epochs=8
num_minibatches=8
partial_reset=1
reconfiguration_freq=1
reward_scale=1.0

# exp data
obs_mode='state_dict'
control_mode='pd_joint_delta_pos'
reward_mode='normalized_dense'
include_force=0
include_state=1
force_encoding="DNN"

date=$(date +"%Y-%m-%d_%H:%M")
#_${date}"

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

for i in $(seq $start $end);
do
    #printf "\n\n\n\nStarting baseline exp ${i}\n\n\n\n"
    #exp_name = "pickcube_state_baseline_" + $i
    exp_name="${exp_set_name}_${i}_${date}"
    python -m learning.ppo \
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
        --exp-name=$exp_name 
done



# store data
#python -m data_analysis.tb_extraction $filepath state $number_of_seeds "runs/${filepath}/"

# store plots
#mkdir -p "runs/${filepath}/plots/"
#python -m data_analysis.pd_graphing "runs/${filepath}/"