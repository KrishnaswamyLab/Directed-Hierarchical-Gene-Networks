for dataset in Cora_ml Texas Cornell Wisconsin; do
    counter=0
    for w in 50 100; do
        for n in 5 10; do
            python train.py --model Node2Vec --walk-length ${w} --num-walks ${n} --save-as ${counter}_${dataset} --dataset ${dataset}
            counter=$((counter+1))
		done
	done
done
