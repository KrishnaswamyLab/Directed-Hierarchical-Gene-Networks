for dataset in Cora_ml Texas Cornell Wisconsin; do
    counter=0
    for lr in 0.01; do
        for c in 0.5 1 1.5; do
            for w in 0.001 0.01; do
                for q in 0.1 0.2 0.0; do
			for J in 5 10 15; do
                    python train.py --model DS-PM --lr ${lr} --c ${c} --act linear --weight-decay ${w} --save-as ${counter} --q ${q} --dataset ${dataset} --J ${J}
                    counter=$((counter+1))
                done        
            done
        done
    done
done
done
