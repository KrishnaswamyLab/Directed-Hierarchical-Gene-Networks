for dataset in Texas Cornell omnipath SIGNOR iPTMnet; do
    counter=0
    for lr in 0.001 0.01 0.1; do
        for bias in 0 1; do
            for dropout in 0 0.25 0.5; do
                for n in 2 3; do
                    for act in relu tanh None; do
                        for w in 0.001 0.01; do
                            python train.py --model GAE --lr ${lr} --bias ${bias} --dropout ${dropout} --num-layers ${n} --act ${act} --weight-decay ${w} --symmetrize-adj 0 --save-as ${counter}_${dataset}_directed --device cuda:0 --dataset ${dataset}
                            counter=$((counter+1))
                        done
                    done
                done
            done
        done
    done
done
