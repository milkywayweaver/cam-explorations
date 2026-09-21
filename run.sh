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
python3 script.py --verbose false > "run_result_$(date +%Y-%m-%d_%H-%M-%S).out"