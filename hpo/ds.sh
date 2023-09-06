for dataset in Cora_ml Texas Cornell Wisconsin; do
    counter=0
    for lr in 0.001 0.01 0.1; do
        for bias in 0 1; do
            for dropout in 0 0.25 0.5; do
                for n in 2 3; do
                    for act in relu tanh None; do
                        for w in 0.001 0.01; do
                            for q in 0.1 0.2 0.0; do
				    for J in 5 10 15; do
                                    python train.py --model DS-AE --lr ${lr} --bias ${bias} --dropout ${dropout} --num-layers ${n} --act ${act} --weight-decay ${w} --save-as ${counter} --q ${q} --dataset ${dataset} --J ${J}
                                counter=$((counter+1))
                            done
                        done
                    done
                done
            done
        done
    done
done
done
