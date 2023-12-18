python /workspace/probing/probing_clean.py \
    --root . \
    --layer_choice 12 \
    --cuda 0 \
    --task "bigram_shift" \
    --bs 1 \
    --num_epochs 10 \
    --lr 0.0001 \
    --model_path "/workspace/model"
    
