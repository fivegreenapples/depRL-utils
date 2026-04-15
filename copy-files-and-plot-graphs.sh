# Stop on any error
set -e
# Echo commands for debugging
set -x

SERVER="$1"
SEED="$2"

BUILD_LOCATION="build"

# Check that SERVER and SEED are not empty
if [ -z "$SERVER" ] || [ -z "$SEED" ]; then
    echo "Error: SERVER and SEED parameters are required"
    echo "Usage: $0 <SERVER> <SEED>"
    exit 1
fi

if [ ! -d "$BUILD_LOCATION" ]; then
    echo "Error: BUILD_LOCATION directory does not exist: $BUILD_LOCATION"
    exit 1
fi

FULL_SEED_PATH=$BUILD_LOCATION/$SEED
# Create all folders for the build products
mkdir -p $FULL_SEED_PATH/checkpoints
mkdir -p $FULL_SEED_PATH/plots/test/normalised
mkdir -p $FULL_SEED_PATH/plots/train/normalised

# Copy config and logs from server
scp $SERVER:~/depRL/$FULL_SEED_PATH/config.yaml ./$FULL_SEED_PATH/
scp $SERVER:~/depRL/$FULL_SEED_PATH/log.csv ./$FULL_SEED_PATH/
scp $SERVER:~/depRL/$FULL_SEED_PATH/script.py ./$FULL_SEED_PATH/

# Generate all plots
"$(dirname "$0")/plot-graphs.sh" "$SEED"

# Copy checkpoints from server
for step in $(seq 5 5 600); do
    STEP_FILE=$FULL_SEED_PATH/checkpoints/step_${step}000000.pt
    if [ ! -f $STEP_FILE ]; then
        scp $SERVER:~/depRL/$STEP_FILE ./$STEP_FILE
    fi
done




