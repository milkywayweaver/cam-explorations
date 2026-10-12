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

models=("FIMFScoreCAM")
backbones=("VGG")
classifiers=("Conv")
projections=("duplicate")
datas=("all")
seeds=(42 16 46 103 209 20 29 3 4 67)
# augmets=("geometric" "zoom" "color" "blur" "erasing")

verbose=false
title="AugmentInvert"
mkdir result-${title}

for model in ${models[@]}; do
    for backbone in ${backbones[@]}; do
        for classifier in ${classifiers[@]}; do
            for projection in ${projections[@]}; do
                for data in ${datas[@]}; do
                    for seed in ${seeds[@]}; do
                        # for i in {1..5}; do
                        echo "Starting run..."
                        # python script.py --augment ${augmets[@]:0:i} --seed ${seed} --model ${model} --backbone ${backbone} --classifier ${classifier} --projection ${projection} --data ${data} --verbose ${verbose} > ./result-${title}/${seed}-${backbone}-${classifier}-${projection}-${data}-${i}.out
                        python script.py --seed ${seed} --model ${model} --backbone ${backbone} --classifier ${classifier} --projection ${projection} --data ${data} --verbose ${verbose} > ./result-${title}/${seed}-${backbone}-${classifier}-${projection}-${data}.out
                        sleep 10
                        # done
                    done
                done
            done
        done
    done
done
