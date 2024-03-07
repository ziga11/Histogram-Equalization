args=1
# Check if one argument was passed
if [ $# != $args ]; then
    >&2 echo "Error: Script takes $args arguments.";
    exit 1;
fi

# Check if file exists
if [ ! -f "$1.cu" ]; then
    >&2 echo "Error: File $1.cu does not exist.";
    exit 2;
fi

# Check if second argument is number
#if ! [[ $2 =~ ^[0-9]+$ ]]; then
   #>&2 echo "Error: $2 is not a number.";
   #exit 3;
#fi

# Check if second argument is zero
#if [ "$2" -eq "0" ]; then
    #>&2 echo "Error: Number of GPUs must be non-zero.";
    #exit 4;
#fi

#path=$(cd $1; pwd)

echo "Loading CUDA module.";
module load CUDA;

echo "Compiling $1.cu to $1.out.";
nvcc "$1.cu" -o "$1.out";

if [ $? -eq 0 ]; then
    echo "Running $1.out:"
    echo "srun --partition=gpu --gpus=1 --mem-per-cpu=32G --ntasks=1 $1.out";
    srun --partition=gpu --gpus=1 --mem-per-cpu=32G --ntasks=1 "$1.out";
else
    echo >&2 "Error: There was an error compiling \"$1.cu\".";
    exit 4;
fi