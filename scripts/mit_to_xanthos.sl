#!/bin/bash

# ------------------------------------------------------------------------------------------
# Convert MIT climate data to the format needed for xanthos. SLURM array
# processes each scenario, model combination per node. There are 126 scenario, model
# combinations (e.g., 0-125). The python code is set to process each realization in
# parallel per node.
#
# TO RUN:
# sbatch --array=0-71 <your_dir_path>/mit_to_xanthos.sl
# ------------------------------------------------------------------------------------------

#SBATCH -n 1
#SBATCH -t 04:00:00
##SBATCH -A mit-pnnl
#SBATCH -J mit-climate
#SBATCH -p fdr

# Load Modules
source /etc/profile.d/modules.sh
module load python/3.9.1

# number of jobs on each node to parallelize
NJOBS=-1

# input climate data directory holding the scenario/model/realization directory structure
CLIMATE_DIR="/net/fs04/d2/xgao/pnnl/d0.5"

# directory to write the outputs to
OUTPUT_DIR="/net/fs04/d2/xgao/pnnl/hd_xanthos"

# xanthos reference file path with filename and extension
XANTHOS_REF_FILE="/net/fs04/d2/xgao/pnnl/hd_xanthos/xanthos_0p5deg_landcell_reference.csv"

start=$(date)
echo "Start:  $start"

python /net/fs04/d2/xgao/pnnl/hd_xanthos/mit_to_xanthos.py $SLURM_ARRAY_TASK_ID $NJOBS $CLIMATE_DIR $OUTPUT_DIR $XANTHOS_REF_FILE

end=$(date)
echo "End:  $end"