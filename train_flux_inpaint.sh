export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux-inpaint"

accelerate launch --config_file accelerate_config.yaml --main_process_port 29600 train_flux_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_inpaint_model_name_or_path="xiaozaa/flux1-fill-dev-diffusers" \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1.5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --validation_epochs=2500 \
  --validation_steps=500 \
  --seed="69" \
  --dataroot="/workspace1/pdawson/tryon-scraping/dataset" \
  --train_data_list="train_pairs.txt" \
  --test_data_list="test_pairs.txt" \
  --train_verification_list="train_verify_pairs.txt" \
  --validation_data_list="verify_pairs.txt"\
  --height=768 \
  --width=576 \
  --max_sequence_length=512  \
  --checkpointing_steps=1000  \
  --report_to="wandb" \
  --train_base_model \
  # --resume_from_checkpoint="latest"  \