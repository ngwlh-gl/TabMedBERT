# KBioXLM是基于XLM-R的，所以其--type=roberta，加载我们的模型需要--reset_weights
```
export LR=2e-5
for SEED in 42
do

python main.py \
    --do_train \
    --data_folder ./datasets/ADE/eng_to_zh \
    --pretrained_dir ../KBioXLM_model \
    --result_filepath ./results/KBioXLM_model-$SEED-$LR.json \
    --max_position_embeddings 512 \
    --output_dir ./ckpts/KBioXLM_model-$SEED-$LR \
    --train_bs 16 \
    --reset_weights \
    --type roberta \
    --do_lower_case \
    --lr $LR \
    --max_epochs 100 \
    --task ade \
    --seed $SEED \

done

```

### pretrained_dir: 模型存放位置
### output_dir: 训练后的模型存放路径
### result_filepath: 最后评测的F1值等的存放文件路径
### data_folder: 数据集路径