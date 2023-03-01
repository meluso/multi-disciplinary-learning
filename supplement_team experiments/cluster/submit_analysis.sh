### Regressions & Random Forests

# Specify log directory
LOGDIR=/gpfs1/home/j/m/jmeluso/isl-group-experiments/logs/analysis

# Make log directory
if [ ! -d $LOGDIR ] ; then
	mkdir -p $LOGDIR
fi

# Submit analysis jobs
sbatch submit_regressions.sbat
sbatch submit_random_forests.sbat