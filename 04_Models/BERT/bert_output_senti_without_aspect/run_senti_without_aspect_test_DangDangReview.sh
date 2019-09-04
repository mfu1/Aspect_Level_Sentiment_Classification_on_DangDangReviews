export BASE_DIR=gs://ddreview-bucket
export DATA_DIR=$BASE_DIR/DangDangReview/data_DangDang_review_for_bert_ascpect_prediction
export BERT_BASE_DIR=$BASE_DIR/BERT/bert_output_senti_without_aspect
export OUTPUT_DIR=$BASE_DIR/DangDangReview/
export TRAINED_CLASSIFIER=$BERT_BASE_DIR/model.ckpt-129
export TPU_NAME=meifu39

python3 run_classifier.py \
--task_name=semeval \
--do_predict=true \
--data_dir=$DATA_DIR/ \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$TRAINED_CLASSIFIER \
--max_seq_length=128 \
--output_dir=$OUTPUT_DIR/ \
--use_tpu=True \
--tpu_name=$TPU_NAME \
--pred_aspect=true \
--have_aspect=false