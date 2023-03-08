counter=0
for lr in 0.001 0.01 0.1; do
	for bias in 0 1; do
		for dropout in 0 0.25 0.5; do
			for n in 2 3; do
				for act in complexrelu None; do
					for q in 0.1 0.2; do
						for w in 0.001 0.01; do
							python train.py --model MagNet --lr ${lr} --bias ${bias} --dropout ${dropout} --num-layers ${n} --act ${act} --q ${q} --weight-decay ${w} --task existence --save-as ${counter}_existence --num-classes 2
						        counter=$((counter+1))
						done
					done
				done
			done
		done
	done
done
