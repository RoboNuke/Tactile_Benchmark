exp_types=("FragilePegInsert-v1")
obs_types=("state_dict" "state_dict_no_ft" "rgb" "rgb_no_ft")
dmg_vals=("25.0" "50.0" "100.0" "250.0" "500.0" "100000.0")


for obs_type in ${obs_types[@]}; do
    for dmg_val in ${dmg_vals[@]}; do 
        bash learning/3_launch_tmux.bash "FragilePegInsert-v1" $obs_type $dmg_val
    done
done