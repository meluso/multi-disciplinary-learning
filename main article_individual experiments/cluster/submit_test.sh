# Specify log directory
LOGDIR=/gpfs1/home/j/m/jmeluso/ci-greedy-agents-base/logs/test

# Make log directory
if [ ! -d $LOGDIR ] ; then
	mkdir -p $LOGDIR
fi

# Submit jobs for this execution
sbatch --export=ii=1234 submit_test.sbat