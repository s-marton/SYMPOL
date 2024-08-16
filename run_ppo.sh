#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <number_of_runs> <command> [<args>...]"
    exit 1
fi

# Extract the number of runs from the first argument
NUM_RUNS=$1
shift

# Extract the command and its arguments
COMMAND="$@"

# Extract and modify the value of --n_trials
TRIALS_ARG="--n_trials"
TRIALS_VALUE=$(echo "$COMMAND" | awk -v arg="$TRIALS_ARG" '{for(i=1;i<=NF;i++) if ($i==arg) print $(i+1)}')

if [ -z "$TRIALS_VALUE" ]; then
    echo "Error: --n_trials argument not found in the command"
    exit 1
fi

# Calculate new trials value
NEW_TRIALS_VALUE=$(($TRIALS_VALUE / $NUM_RUNS))

# Replace the old trials value with the new one in the command
COMMAND=$(echo "$COMMAND" | sed "s/$TRIALS_ARG $TRIALS_VALUE/$TRIALS_ARG $NEW_TRIALS_VALUE/")

# Print the modified command
echo "Running command: $COMMAND"

# Run the command the specified number of times
for i in $(seq 1 $NUM_RUNS)
do
    sleep 2
    nohup $COMMAND &
done
