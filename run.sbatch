#!/bin/bash
#SBATCH --partition=k80_q
#SBATCH --job-name=dagan
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --account=cs6824_f19

# module load gcc openmpi cuda
module load Anaconda
module load cuda/8.0.44
module load cudnn/5.1
source activate bio

pycommand="/home/wanyef/.conda/envs/bio/bin/python ./ffdagan.py"

echo "Start time: $(date)"
for dataset in "ALLAML" "colon" "GLIOMA" "leukemia" "lung_discrete" "lung" "lymphoma" "Prostate_GE" "TOX_171"
# skipping these because too big to run atm. for dataset in "CLL_SUB_111" "GLI_85" "nci9" "SMK_CAN_187"
do
    ${pycommand} -d ${dataset}
done
echo "End time: $(date)"

