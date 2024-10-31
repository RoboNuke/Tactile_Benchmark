task="SimpleFragilePiH-v1"
#task="FragilePegInsert-v1"
#task="ForeignObjectRemoval-v1"
num_steps=150
total_timesteps=30000000

obs_types=("state_dict_no_ft") # "state_dict") # "rgb" "rgb_no_ft")
dmg_vals=("100000.0") # "50.0" "100.0" "250.0" "500.0" ) 
control_modes=("pd_joint_delta_pos")
reward_modes=("normalized_dense")
force_encodings=("FFN")
#critic_n=("1" "1" "1" "2" "3")
#critic_l=("128" "256" "512" "512" "1024")
critic_n=("2")
critic_l=("512")
use_shampoo=0
lock_gripper=1

ent_coeffs=("0.0001" "0.00005" "0.00001" "0.000005")
#ent_co="0.0"
#lrs=("0.0003" "0.003" "0.03")

tmux new-session -d -s "Holder"
for control_mode in ${control_modes[@]}; do
    for reward_mode in ${reward_modes[@]}; do
        for force_encoding in ${force_encodings[@]}; do
            for obs_type in ${obs_types[@]}; do
                if [[ $obs_type == *"no_ft"* ]]; then
                    f_code="no-ft"
                else
                    f_code="ft"
                fi
                for dmg_val in ${dmg_vals[@]}; do
		            #echo ${dmg_val} ${obs_type}
		            #for (( i=0; i<4; i++ )); do	
                    #echo "    " ${critic_n[$i]} ${critic_l[$i]}
                    if [[ $dmg_val == "100000.0" ]]; then
                        dmg_code="None"
                    else
                        dmg_code=$dmg_val
                    fi
                    for ent_co in ${ent_coeffs[@]}; do
                    #for lr in ${lrs[@]}; do

		                exp_name="EntTest_(${ent_co})" # "Base_SFPiH_(${f_code})_($dmg_code)"
                        
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
                            "512" \
                            $use_shampoo \
                            $lock_gripper \
                            $ent_co #\
                            #$lr
                            # ${critic_l[0]} 
		            done	
                done
            done
        done
    done
done
