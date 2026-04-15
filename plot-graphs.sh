# Stop on any error
set -e
# Echo commands for debugging
set -x

SEED="$1"

BUILD_LOCATION="build"

# Check that BUILD_LOCATION, NAME and SEED are not empty
if [ -z "$SEED" ]; then
    echo "Error: SEED parameter is required"
    echo "Usage: $0 <SEED>"
    exit 1
fi

if [ ! -d "$BUILD_LOCATION" ]; then
    echo "Error: BUILD_LOCATION directory does not exist: $BUILD_LOCATION"
    exit 1
fi

FULL_SEED_PATH=$BUILD_LOCATION/$SEED

# Generate all plots
python3 depRL-utils/plot_all_stats.py        --path $FULL_SEED_PATH
python3 depRL-utils/plot_training_diagnostics.py    $FULL_SEED_PATH
python3 depRL-utils/plot_rewards.py                 $FULL_SEED_PATH
python3 depRL-utils/plot_rewards_stacked.py         $FULL_SEED_PATH
python3 depRL-utils/plot_rewards_stacked_normed.py  $FULL_SEED_PATH
