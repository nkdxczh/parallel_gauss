TARGET=./gauss

for s in 32 64 128 256 300 512 700 1024 1200 1500 1700 2048 2500 3000 4096 8192
do
    echo $s
    for i in `seq 5`
    do
        $TARGET -n $s
        $TARGET -p -n $s
    done
done
