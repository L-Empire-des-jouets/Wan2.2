# Résumé des corrections pour l'erreur OOM avec Ulysses

## Problème identifié

Lors de l'utilisation de 2 GPU avec Ulysses, le décodage VAE provoque un OOM car :
1. Seul le GPU 0 (rank 0) effectue le décodage
2. Toutes les données doivent être rassemblées sur un seul GPU
3. Le chunk_t par défaut (20) est trop élevé pour cette configuration

## Modifications apportées

### 1. Détection automatique multi-GPU (wan/textimage2video.py)
- Ajout de la détection du nombre de GPU pour ajuster automatiquement chunk_t
- Valeur par défaut : 8 pour multi-GPU, 20 pour single GPU
- Synchronisation et nettoyage mémoire avant le décodage

### 2. Gestion mémoire améliorée dans _decode_in_chunks
- Détection de la mémoire GPU disponible
- Réduction automatique de chunk_t si mémoire faible (<5GB)
- Option de décodage sur CPU si mémoire critique (<3GB)
- Nettoyage agressif de la mémoire après chaque chunk

## Configuration recommandée

Pour votre configuration (2x RTX 5090 32GB), utilisez :

```bash
# Configuration mémoire optimisée
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Chunk size réduite (optionnel, le code ajuste automatiquement)
export WAN_VAE_DECODE_CHUNK_T=8

# Si encore des problèmes, essayez :
export WAN_VAE_DECODE_CHUNK_T=6
# ou même
export WAN_VAE_DECODE_CHUNK_T=4
```

## Script de test

Un script `test_fix.sh` a été créé avec la configuration optimale.

## Solutions alternatives si le problème persiste

1. **Réduire encore chunk_t** : `export WAN_VAE_DECODE_CHUNK_T=4`

2. **Désactiver P2P si problèmes** : `export NCCL_P2P_DISABLE=1`

3. **Solution ultime** : Modifier le code pour faire un décodage distribué où chaque GPU décode sa partie (nécessite des changements plus importants)

## Impact sur les performances

- Le temps de décodage sera légèrement plus long avec des chunks plus petits
- Mais cela permet d'utiliser Ulysses qui divise par 2 le temps de génération total
- Le gain net reste positif

## Commande de test

```bash
./test_fix.sh
```

Ou directement :

```bash
export WAN_VAE_DECODE_CHUNK_T=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
torchrun --standalone --nproc_per_node=2 generate.py \
  --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --dit_fsdp --ulysses_size 2 \
  --offload_model True --convert_model_dtype --t5_cpu \
  --frame_num 121 --sample_steps 50 \
  --save_file ./results/result_2_gpu_$(date +%F_%H-%M-%S).mp4 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```