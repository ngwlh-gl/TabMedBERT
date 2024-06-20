export CUDA_VISIBLE_DEVICES=0


python run_relation.py \
  --task cdr \
  --train_file /data1/gl/project/ner-relation/ner/data/official_CDR/json/train.json \
  --do_train --do_eval --eval_test \
  --model /data1/gl/project/ner-relation/pretrained/tabmedbert/tabmedbert-linkbert-large \
  --do_lower_case \
  --context_window 0 \
  --max_seq_length 512 \
  --entity_output_dir /data1/gl/project/ner-relation/ner/data/official_CDR/predicted_json/tabmedbert-linkbert-large \
  --output_dir /data1/gl/project/ner-relation/PURE-main/ckpts/relation/gold/tabmedbert-linkbert-large/cdr \
  --entity_predictions_dev dev.json \
  --entity_predictions_test test.json \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --train_batch_size 8


