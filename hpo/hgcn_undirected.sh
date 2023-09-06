for dataset in Cora_ml Texas Cornell Wisconsin; do
counter=0
for lr in 0.001 0.01 0.1; do
	for bias in 0 1; do
		for dropout in 0 0.25 0.5; do
			for act in relu tanh None; do
                for n in 2 3; do
                    for w in 0.001 0.01; do
                        for c in 1 None; do
                            python hyperbolic_methods/train.py --task lp --model HGCN --manifold PoincareBall --lr ${lr} --weight-decay ${w} --num-layers ${n} --dropout ${dropout} --act ${act} --bias ${bias} --optimizer Adam --c ${c} --symmetrize 1 --save 1 --save-as ${counter}_${dataset}_undirected --cuda 1 --dataset ${dataset}
                            counter=$((counter+1))
                        done
                    done
				done
			done
		done
	done
done
done
