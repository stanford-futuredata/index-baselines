if [ "$#" -ne 1 ]; then
    echo "Usage: run_hashing.sh PATH"
    exit 1
fi

echo "Running hashing baselines"
./hashing "$1"
