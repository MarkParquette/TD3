#!/bin/bash

# Generate baseline results using the original TD3 algorithm

for ((i=9;i<10;i+=1))
do 
	python main.py \
	--policy "TD3-DEV" \
	--env "LunarLanderContinuous-v3" \
	--dev \
	--seed $i
done
