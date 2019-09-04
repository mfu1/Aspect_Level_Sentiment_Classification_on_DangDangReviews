export BASE_DIR=gs://ddreview3-bucket
export DATA_DIR=$BASE_DIR/Finetune_Aspect_on_TextRank
export BERT_BASE_DIR=$BASE_DIR/BERT/bert_output_aspect
export OUTPUT_DIR=$BASE_DIR/Transfer/Finetuned_Aspect_Model
export TRAINED_CLASSIFIER=$BERT_BASE_DIR/model.ckpt-129
export TPU_NAME=mf19690129

python3 run_classifier.py \
--task_name=semeval \
--do_train=True \
--do_eval=true \
--data_dir=$DATA_DIR/ \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$TRAINED_CLASSIFIER \
--max_seq_length=128 \
--train_batch_size=320 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=$OUTPUT_DIR/ \
--use_tpu=True \
--tpu_name=$TPU_NAME \
--pred_aspect=True \
--have_aspect=False