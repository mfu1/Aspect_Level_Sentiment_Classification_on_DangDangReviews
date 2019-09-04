export BASE_DIR=gs://ddreview3-bucket
export DATA_DIR=$BASE_DIR/Finetune_Aspect_on_TextRank
export BERT_BASE_DIR=$BASE_DIR/Finetuned_Aspect_Model
export OUTPUT_DIR=$BASE_DIR/DangDangReview
export TRAINED_CLASSIFIER=$BERT_BASE_DIR/model.ckpt-19834
export TPU_NAME=mfu19690129

python3 run_classifier_DDR.py \
--task_name=semeval \
--do_predict=true \
--data_dir=$DATA_DIR/ \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$TRAINED_CLASSIFIER \
--max_seq_length=128 \
--output_dir=$OUTPUT_DIR/ \
--use_tpu=False \
--pred_aspect=true \
--have_aspect=false
