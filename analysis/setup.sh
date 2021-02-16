#!/bin/bash
overlay_ext3=/scratch/ksf293/anomalies/overlay-15GB-500K.ext3:ro
singularity \
exec --overlay $overlay_ext3:ro \
/scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash \
-c "source /ext3/env.sh; \
conda activate anomenv; \
"
