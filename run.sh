#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00

cd $PWD
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/labupk/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
eval "$__conda_setup"
else
if [ -f "/home/labupk/miniconda3/etc/profile.d/conda.sh" ]; then
. "/home/labupk/miniconda3/etc/profile.d/conda.sh"
else
export PATH="/home/labupk/miniconda3/bin:$PATH"
fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate torch
python3 script.py > run_result.out