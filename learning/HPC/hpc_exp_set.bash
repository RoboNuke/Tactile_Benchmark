task="FragilePegInsert-v1"
num_steps=150
total_timesteps=50000000

obs_types=("state_dict_no_ft" "state_dict") # "rgb" "rgb_no_ft")
dmg_vals=("500.0" "100000.0") # "100.0" "50.0" "250.0" "100000.0") 
control_modes=("pd_joint_delta_pos")
reward_modes=("normalized_dense")
force_encodings=("FFN")
#critic_n=("1" "1" "1" "2" "3")
#critic_l=("128" "256" "512" "512" "1024")
critic_n=("2")
critic_l=("512")

tmux new-session -d -s "Holder"
for control_mode in ${control_modes[@]}; do
    for reward_mode in ${reward_modes[@]}; do
        for force_encoding in ${force_encodings[@]}; do
            for obs_type in ${obs_types[@]}; do
                for dmg_val in ${dmg_vals[@]}; do
		            echo ${dmg_val} ${obs_type}
		            #for (( i=0; i<5; i++ )); do	
                    #echo "    " ${critic_n[$i]} ${critic_l[$i]}
		                exp_name="setting_${dmg_val}_${obs_type}"
                        sbatch learning/HPC/hpc_sbach_cmd.bash \
                            $task \
                            $obs_type \
                            $dmg_val \
                            $exp_name \
                            $num_steps \
                            $total_timesteps \
                            $control_mode \
                            $reward_mode \
                            $force_encoding \
                            "2" \
                            "512" 
                            # ${critic_l[0]} 
		            #done	
                done
            done
        done
    done
done
