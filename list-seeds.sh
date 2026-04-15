# Stop on any error
set -e

# Check that at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Error: At least one server parameter is required"
    echo "Usage: $0 <SERVER> [SERVER2] [SERVER3] ..."
    exit 1
fi


for SERVER in "$@"; do
    echo $SERVER
    ssh "$SERVER" "find depRL/build -mindepth 3 -maxdepth 3 -type d -printf '$SERVER %P\n'"
    echo
done