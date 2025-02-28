export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-viton"

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file accelerate_config.yaml --main_process_port 29700 train_flux_txt2img.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --validation_epochs=2500 \
  --validation_steps=500 \
  --seed="69" \
  --dataroot="/workspace1/pdawson/tryon-scraping/vitonhd" \
  --train_data_list="train_pairs.txt" \
  --train_verification_list="verification_train.txt" \
  --validation_data_list="verification_test.txt"\
  --height=768 \
  --width=576 \
  --max_sequence_length=512  \
  --checkpointing_steps=1000  \
  --report_to="wandb" \
  --train_base_model \
