#!/bin/bash

# Generate baseline results using the original TD3 algorithm

for ((i=0;i<10;i+=1))
do 
	python main.py \
	--policy "TD3" \
	--env "LunarLanderContinuous-v3" \
	--seed $i
done
