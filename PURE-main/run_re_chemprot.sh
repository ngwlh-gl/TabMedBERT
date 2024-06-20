
export CUDA_VISIBLE_DEVICES=2

for SEED in 52
do

python run_relation.py \
  --task chemprot \
  --train_file /data1/gl/project/ner-relation/LinkBERT/data/seqcls/chemprot_hf/json/train.json \
  --do_train --do_eval --eval_test --eval_with_gold \
  --model microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract \
  --do_lower_case \
  --context_window 0 \
  --max_seq_length 256 \
  --entity_output_dir /data1/gl/project/ner-relation/LinkBERT/data/seqcls/chemprot_hf/json \
  --output_dir /data1/gl/project/ner-relation/PURE-main/ckpts/relation/gold/PubMedBERT-large/$SEED-chemprot \
  --entity_predictions_dev dev.json \
  --entity_predictions_test test.json \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --train_batch_size 16 \
  --seed $SEED

done
