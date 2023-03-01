### Regressions & Random Forests

# Specify log directory
LOGDIR=/gpfs1/home/j/m/jmeluso/ci-greedy-agents-base/logs/analysis

# Make log directory
if [ ! -d $LOGDIR ] ; then
	mkdir -p $LOGDIR
fi

# Submit analysis jobs
sbatch submit_regressions.sbat
sbatch submit_random_forests.sbat