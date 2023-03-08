counter=0
for lr in 0.001 0.01 0.1; do
	for bias in 0 1; do
		for dropout in 0 0.25 0.5; do
			for n in 2 3; do
				for act in relu tanh None; do
					for w in 0.001 0.01; do
						python train.py --model UDS-AE --lr ${lr} --bias ${bias} --dropout ${dropout} --num-layers ${n} --act ${act} --weight-decay ${w} --save-as ${counter}
						counter=$((counter+1))
                    			done
				done
			done
		done
	done
done
