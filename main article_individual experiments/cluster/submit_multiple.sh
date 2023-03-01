### Executions 45-48

for EXECNUM in $(seq 45 48)
do
	# Specify execution number
	EXECFMT=$(printf "%03d" $EXECNUM)

	# Specify log and data directorires
	LOGDIR=/gpfs1/home/j/m/jmeluso/ci-greedy-agents-base/logs/exec$EXECFMT

	# Make log directory
	if [ ! -d $LOGDIR ] ; then
		mkdir -p $LOGDIR
	fi

	# Submit jobs for this execution
	for i in $(seq 0 249)
	do
		sbatch --export=ii=${i} submit_exec$EXECFMT.sbat
		sleep 0.5
	done

done