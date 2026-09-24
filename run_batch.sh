#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00

cd $PWD
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/radithya/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
eval "$__conda_setup"
else
if [ -f "/home/radithya/anaconda3/etc/profile.d/conda.sh" ]; then
. "/home/radithya/anaconda3/etc/profile.d/conda.sh"
else
export PATH="/home/radithya/anaconda3/bin:$PATH"
fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate torch

export CUBLAS_WORKSPACE_CONFIG=:4096:8

model="GradCAM"
backbones=("VGG")
classifiers=("Conv")
projections=("duplicate")
datas=("all")
seeds=(16 42 46 67 29 20 4 3 103 209)

verbose=false
title="Seed"
mkdir result-${title}

for backbone in ${backbones[@]}; do
    for classifier in ${classifiers[@]}; do
        for projection in ${projections[@]}; do
            for data in ${datas[@]}; do
		        for seed in ${seeds[@]}; do
                    echo "Starting run for seed ${seed}, ${backbone} backbone, ${classifier} classifier, and ${projection} projection with ${data}."
                    python script.py --exp_name ${title} --seed ${seed} --model ${model} --backbone ${backbone} --classifier ${classifier} --projection ${projection} --data ${data} --verbose ${verbose} > ./result-${model}/${backbone}-${classifier}-${projection}-${data}.out
                    sleep 10
                done
            done
        done
    done
done
