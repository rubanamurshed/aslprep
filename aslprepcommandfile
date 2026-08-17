#!/bin/bash
#SBATCH --job-name=ASLprep_sub4028         # Job name shown in queue — "sub4028" here is just an example/placeholder subject, edit this per run
#SBATCH --output=/data/raulab/asl2/output/logs/aslprep_%j.out   # Standard output log, %j = job ID
#SBATCH --error=/data/raulab/asl2/output/logs/aslprep_%j.err    # Standard error log, %j = job ID
#SBATCH --nodes=1                          # Run on a single compute node
#SBATCH --ntasks=1                         # One task (ASLPrep isn't MPI-parallel, so no need for more)
#SBATCH --cpus-per-task=16                 # Allocate 16 CPU cores for this task
#SBATCH --time=10:00:00                    # Max walltime: 10 hours before SLURM kills the job
#SBATCH --mail-user=rubana.mursh@gmail.com # Email notifications go here
#SBATCH --mail-type=FAIL                   # Only email on job failure (not on start/completion)

## Reminder for how to actually launch this job:
## cd /data/raulab/asl2/output/logs #file path,from raucher lab on crisp database
## sbatch /data/raulab/asl2/aslprep_rubana.sh

# --- Environment Variables ---
export FS_LICENSE=/data/raulab/asl2/config/license.txt                  # Path to FreeSurfer license on the host system
export TEMPLATEFLOW_HOME=/data/raulab/asl2/config/templateflow          # Host-side cache of TemplateFlow atlases/templates
export APPTAINERENV_FS_LICENSE=/config/license.txt                      # Same license path, but as seen INSIDE the container
export APPTAINERENV_TEMPLATEFLOW_HOME=/templateflow                     # Same TemplateFlow path, but as seen INSIDE the container
#export TMPDIR=/data/raulab/asl2/asltmp                                 # (Unused) alternate temp dir, left commented as a reference

# Node-local work directory — where ASLPrep writes intermediate/scratch files during the run
WORKDIR=/data/raulab/asl2/aslwd 

echo "Using workdir: ${WORKDIR}"

# --- Run ASLPrep inside the Apptainer/Singularity container ---
apptainer run --cleanenv \
  --env TEMPLATEFLOW_HOME=/templateflow \
  --bind /data/raulab/asl2/my_dataset:/bids:ro \        # Mount BIDS input data read-only at /bids
  --bind /data/raulab/asl2/output:/out \                # Mount output directory at /out
  --bind /data/raulab/asl2/config:/freesurfer_license \ # Mount config folder (contains FS license) at /freesurfer_license
  --bind /data/raulab/asl2/config/templateflow:/templateflow \  # Mount TemplateFlow cache at /templateflow
  --bind ${WORKDIR}:/work \                             # Mount scratch/work directory at /work
  /data/raulab/asl2/aslprep-25.1.0.sif \                # The ASLPrep v25.1.0 container image
  /bids /out participant \                              # ASLPrep positional args: input dir, output dir, analysis level
  --participant-label 5012\                             # Subject to process — CHANGE THIS per run (this example was actually run on 5012, not 4028)
  --smooth-kernel 5 \                                   # Spatial smoothing kernel (mm) applied to CBF maps
  --skip_bids_validation \                              # Skip strict BIDS validation (data already cleaned/checked)
  --output-spaces T1w MNI152NLin2009aSym \              # Output CBF maps in both native T1w and standard MNI space
  --random-seed 42 \                                    # Fixed seed for reproducibility
  --n_cpus 16 \                                         # Number of CPUs ASLPrep itself should use (matches --cpus-per-task above)
  --basil \                                              # Use BASIL toolbox for CBF quantification (FSL's Bayesian ASL model)
  --skull-strip-fixed-seed \                            # Use a fixed random seed for skull-stripping step too
  --fs-license-file /freesurfer_license/license.txt \   # Path to FreeSurfer license (inside container)
  --work-dir /work \                                    # Working/scratch directory (inside container)
  --clean-workdir                                        # Delete intermediate work-dir contents after successful completion
