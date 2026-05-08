#! /bin/bash
export MLFLOW_TRACKING_URI=$3

# Get current timestamp for experiment naming
printf -v now '%(%F_%H-%M-%S)T' -1
exp_name="tsc_deep_$1_$now"
exp_create_str=$(mlflow experiments create -n $exp_name)
echo $exp_create_str
exp_id=$(echo $exp_create_str | awk '{print $NF}')

models=("ConvTran" "FCN" "InceptionTime" "LSTMFCN" "MCDCNN" "MLP" "ResNet")
datasets=("Pedestrian" "WISDM" "UCIActivity" "LakeIce" "Tiselac" "InsectSound" "USCActivity" "FordChallenge" "CrowdSourced" "WISDM2" "Skoda" "STEW" "AudioMNIST-DS" "CornellWhaleChallenge" "FruitFlies" "Opportunity" "PAMAP2" "WhaleSounds" "DREAMERA" "DREAMERV")

for d in "${datasets[@]}"
do
    for m in "${models[@]}"
    do
        for bs in "16" "32" "64" "128" "256" "512" "1024" "2048"
        do
            for f in "0" "1" "2" "3" "4"
            do
                echo "Running model $m on GPU $1 on DS $d for fold $f ..."
                mlflow run --experiment-name=$exp_name -e main.py -P gpu=$1 -P cache_dir=$2 -P use_pretrained=$4 -P dataset=$d -P model=$m -P fold=$f -P batch_size=$bs -P seed=-1 ./tsc
            done
        done
        # Save experiment data to CSV
        mlflow experiments csv -x $exp_id > "$exp_name.csv"
    done
done