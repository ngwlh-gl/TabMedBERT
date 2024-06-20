
export CUDA_VISIBLE_DEVICES=0

export LR=2e-5
for SEED in 42
do

python main.py \
    --do_train \
    --data_folder ./data/ADE-10-folders/0/txt \
    --pretrained_dir ../tabmedbert/tabmedbert-linkbert-large \
    --result_filepath ./results/tabmedbert-$SEED-$LR.json \
    --max_position_embeddings 512 \
    --output_dir ./ckpts/tabmedbert-$SEED-$LR \
    --train_bs 16 \
    --type bert \
    --do_lower_case \
    --lr $LR \
    --max_epochs 100 \
    --task ade \
    --seed $SEED \

done
    

