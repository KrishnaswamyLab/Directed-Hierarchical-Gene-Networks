counter=0
for lr in 0.001 0.01 0.1; do
	for temp in 0.5 1.0; do
		for margin in 3.0 9.0; do
			for w in 0.001 0.01; do
				python train.py --model TransE --lr ${lr} --temperature ${temp} --margin ${margin} --weight-decay ${w} --edge_attribute 0 --save-as ${counter}_noedgeatt
                counter=$((counter+1))
			done
		done
	done
done
