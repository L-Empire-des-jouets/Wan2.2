#!/bin/bash
# Script de test pour la correction OOM avec Ulysses

echo "Test de la génération vidéo avec 2 GPU et les corrections OOM"
echo "============================================================"

# Configuration optimisée pour 2x RTX 5090 (32GB chacune)
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29512
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# Configuration mémoire optimisée
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Chunk size réduite pour multi-GPU (sera override par le code si nécessaire)
export WAN_VAE_DECODE_CHUNK_T=8

echo "Configuration:"
echo "- GPUs: $CUDA_VISIBLE_DEVICES"
echo "- WAN_VAE_DECODE_CHUNK_T: $WAN_VAE_DECODE_CHUNK_T"
echo "- PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Commande de génération
torchrun --standalone --nproc_per_node=2 generate.py \
  --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --dit_fsdp --ulysses_size 2 \
  --offload_model True --convert_model_dtype --t5_cpu \
  --frame_num 121 --sample_steps 50 \
  --save_file ./results/result_2_gpu_$(date +%F_%H-%M-%S).mp4 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."