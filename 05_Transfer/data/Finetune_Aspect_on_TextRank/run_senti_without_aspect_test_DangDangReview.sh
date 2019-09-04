export BASE_DIR=gs://ddreview3-bucket
export DATA_DIR=$BASE_DIR/Finetune_Aspect_on_TextRank
export BERT_BASE_DIR=$BASE_DIR/bert_output_senti_without_aspect
export OUTPUT_DIR=$BASE_DIR/DangDangReview
export TRAINED_CLASSIFIER=$BERT_BASE_DIR/model.ckpt-129

python3 run_classifier_DDR.py \
--task_name=semeval \
--do_predict=True \
--data_dir=$DATA_DIR/ \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$TRAINED_CLASSIFIER \
--max_seq_length=128 \
--output_dir=$OUTPUT_DIR/ \
--use_tpu=False \
--pred_aspect=False \
--have_aspect=False