for dataset in Texas Cornell omnipath SIGNOR iPTMnet; do
    counter=0
    for lr in 0.001 0.01 0.1; do
        for bias in 0 1; do
            for dropout in 0 0.25 0.5; do
                for act in relu tanh None; do
                    for w in 0.001 0.01; do
                        python hyperbolic_methods/train.py --task lp --model Shallow --manifold PoincareBall --lr ${lr} --weight-decay ${w} --num-layers 0 --use-feats 0 --dropout ${dropout} --act ${act} --bias ${bias} --optimizer RiemannianAdam --symmetrize 0 --save 1 --save-as ${counter}_${dataset}_directed --cuda 0 --dataset ${dataset} --epochs 50
                        counter=$((counter+1))
                    done
                done
            done
        done
    done
done
