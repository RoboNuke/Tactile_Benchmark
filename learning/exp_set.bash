exp_types=("FragilePegInsert-v1")
obs_types=("state_dict") # "state_dict_no_ft")
dmg_vals=("0" "25") # "50" "100" "250" "500" "1000")


for obs_type in ${obs_types[@]}; do
    for dmg_val in ${dmg_vals[@]}; do 
        bash learning/3_launch_tmux.bash "FragilePegInsert-v1" $obs_type $dmg_val
    done
done