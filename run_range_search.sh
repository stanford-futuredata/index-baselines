if [ "$#" -ne 1 ]; then
    echo "Usage: run_range_search.sh PATH"
    exit 1
fi
NUM_TRIALS=1
CHUNK_SIZE=64

echo "Running range search baselines"
./range_search "$1" $NUM_TRIALS $CHUNK_SIZE $CHUNK_SIZE $CHUNK_SIZE

echo ""
echo "Running FAST baseline"
./fast "$1"
