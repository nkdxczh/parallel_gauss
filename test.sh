TARGET=./gauss

for s in 256 512 1024 2048 4096 8192
do
    echo "Running at size=" $s
    for i in `seq 5`
    do
        $TARGET -n $s
    done
done
