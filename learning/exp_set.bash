exp_types=("FragilePegInsert-v1")
obs_types=("state_dict" "state_dict_no_ft") # "rgb" "rgb_no_ft")
dmg_vals=("500.0" "100000.0" "100.0" "50" "250.0" )


tmux new-session -d -s "Holder"
for dmg_val in ${dmg_vals[@]}; do 
    for obs_type in ${obs_types[@]}; do
        echo $obs_type $dmg_val
        if [ $obs_type == state_dict ] & [ $dmg_val == 25 ]; then
            echo "    skipping"
        else
            echo '    starting'
            exp_name="FPiH_peg_dmg_${dmg_val}"
            bash learning/3_launch_tmux.bash "FragilePegInsert-v1" $obs_type $dmg_val $exp_name
            echo '    complete'
        fi
    done
done