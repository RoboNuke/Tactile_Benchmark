task="FragilePegInsert-v1"
num_steps=150
total_timesteps=25000000

obs_types=("state_dict") # "state_dict_no_ft") # "rgb" "rgb_no_ft")
dmg_vals=("500.0" "100000") # "100.0" "50.0" "250.0" "100000.0") 
control_modes=("pd_joint_delta_pos")
reward_modes=("normalized_dense")
force_encodings=("FFN")

tmux new-session -d -s "Holder"
for control_mode in ${control_modes[@]}; do
    for reward_mode in ${reward_modes[@]}; do
        for force_encoding in ${force_encodings[@]}; do
            for obs_type in ${obs_types[@]}; do
                for dmg_val in ${dmg_vals[@]}; do 
                    echo $obs_type $dmg_val
                    exp_name="FPiH_peg_dmg_${dmg_val}"
                    sbatch learning/HPC/hpc_sbach_cmd.bash \
                        $task \
                        $obs_type \
                        $dmg_val \
                        $exp_name \
                        $num_steps \
                        $total_timesteps \
                        $control_mode \
                        $reward_mode \
                        $force_encoding 
                done
            done
        done
    done
done
