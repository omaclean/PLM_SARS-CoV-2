# Define target tracking arrays containing paths to OG architectures and local fine-tuned checkpoints
MODELS=(

    "/home3/oml4h/hugging_face_downloads/ESM_C_300M"
    "/home3/oml4h/hugging_face_downloads/ESM_C_600M"
    "/home3/oml4h/hugging_face_downloads/ESM_C_6B"
    "/home3/oml4h/my_SC2_finetunes/myflu/fulldatasetESMcMagma/model/final_checkpoint/"
    "facebook/esm2_t33_650M_UR50D" 
  
)

for MODEL_PATH in "${MODELS[@]}"; do
  echo "========================================================================="
  echo "Processing: $MODEL_PATH"
  echo "========================================================================="
  python3  /home3/oml4h/PLM_SARS-CoV-2/Notebooks/OM_influenza/model_plant_predictions.py \
    --input_csv "/home3/oml4h/PLM_SARS-CoV-2/Results/PLANT_results/PLANT_input_filtered.csv" \
    --coords /home3/oml4h/PLM_SARS-CoV-2/Results/PLANT_results/PLANT_embeddings_with_distance.csv \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "/home3/oml4h/PLM_SARS-CoV-2/Results/PLANT_results/Layer_Analysis" \
    --batch_size 32
done