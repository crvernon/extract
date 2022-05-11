#!/bin/bash

# ------------------------------------------------------------------------------------------
# Convert MIT climate data to the format needed for xanthos. SLURM array
# processes each scenario, model combination per node.
#
# TO RUN:
# sbatch --array=0-126 /rcfs/projects/gcims/projects/mit_climate/code/run_mit_to_xanthos.sl
# ------------------------------------------------------------------------------------------

#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A IHESD
#SBATCH -J mit-climate
#SBATCH -p short

# Load Modules
module purge
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh

# number of jobs on each node to parallelize
NJOBS=-1

# input climate data directory holding the scenario/model/realization directory structure
CLIMATE_DIR="/rcfs/projects/gcims/projects/mit_climate/temp"

# directory to write the outputs to
OUTPUT_DIR="/rcfs/projects/gcims/projects/mit_climate/data/outputs"

# xanthos reference file path with filename and extension
XANTHOS_REF_FILE="/rcfs/projects/gcims/projects/mit_climate/data/reference/xanthos_0p5deg_landcell_reference.csv"

start=$(date)
echo "Start:  $start"

python /rcfs/projects/gcims/projects/mit_climate/code/mit_to_xanthos.py $SLURM_ARRAY_TASK_ID $NJOBS $CLIMATE_DIR $OUTPUT_DIR $XANTHOS_REF_FILE

end=$(date)
echo "End:  $end"