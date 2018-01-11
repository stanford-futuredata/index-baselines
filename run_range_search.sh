if [ "$#" -ne 1 ]; then
    echo "Usage: run_range_search.sh PATH"
    exit 1
fi

NUM_TRIALS=1

echo "Running range search baselines"
./range_search "$1" $NUM_TRIALS 48 48 48

echo ""
echo "Running FAST baseline"
./fast "$1"
