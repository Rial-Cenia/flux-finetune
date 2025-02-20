export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="trained-flux-lora"

CUDA_VISIBLE_DEVICES=6,7 accelerate launch train_flux_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataroot="/workspace1/pdawson/tryon-scraping/dataset2" \
  --train_data_list="train_pairs.txt" \
  --train_verification_list="train_verify_pairs.txt" \
  --validation_data_list="selected_test_pairs.txt"\
  --dataset_name=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --height=768 \
  --width=512 \
  --train_batch_size=3 \
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy"\
  --train_text_encoder\
  --lr_scheduler="constant" \
  --guidance_scale=1.0 \
  --lr_warmup_steps=0 \
  --rank=32 \
  --max_train_steps=50000 \
  --checkpointing_steps=1000 \
  --validation_steps=500 \
  --seed="69" \
  --push_to_hub