#!/usr/bin/env python3
"""
Solutions pour résoudre le problème OOM lors du décodage VAE avec Ulysses
"""

# Solution 1: Augmenter le chunking temporel et optimiser la mémoire
# Modifier dans wan/textimage2video.py, fonction _decode_in_chunks

def _decode_in_chunks_optimized(self, z, chunk_t=8):
    """
    Decode latents in temporal chunks to reduce memory usage.
    Version optimisée avec meilleure gestion mémoire
    """
    # z: [C, T, H', W']
    T = z.shape[1]
    decoded_parts = []
    
    # Forcer la libération de la mémoire avant le décodage
    if hasattr(self, 'model'):
        self.model.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    for i in range(0, T, chunk_t):
        part = z[:, i:i+chunk_t]    # [C, t, H', W']
        
        # Déplacer le chunk sur le GPU seulement quand nécessaire
        if part.device.type == 'cpu':
            part = part.to(self.device)
            
        # Wan2_2_VAE.decode attend une liste de latents; renvoie [video_tensor]
        decoded = self.vae.decode([part])[0]  # [C, (t*H), W]
        
        # Déplacer immédiatement sur CPU pour libérer la VRAM
        decoded = decoded.cpu()
        decoded_parts.append(decoded)
        
        # Libération aggressive de la mémoire après chaque chunk
        del part
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
    # Concaténer sur CPU puis déplacer le résultat final sur GPU si nécessaire
    video = torch.cat(decoded_parts, dim=1)   # [C, (T*H), W]
    
    # Nettoyer
    del decoded_parts
    torch.cuda.empty_cache()
    
    return [video.to(self.device)]


# Solution 2: Décoder de manière distribuée avec all_gather après décodage partiel
def t2v_distributed_decode(self, ...):
    """
    Version modifiée de t2v avec décodage distribué
    """
    # ... code de génération existant ...
    
    # Au lieu de if self.rank == 0:
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Chaque GPU décode sa partie des latents
        x0_local = x0  # Les latents locaux de ce GPU
        
        # Déterminer la taille de chunk adaptée à la mémoire disponible
        chunk_t = int(os.environ.get("WAN_VAE_DECODE_CHUNK_T", "6"))  # Plus petit pour 2 GPU
        
        # Décoder localement
        videos_local = self._decode_in_chunks(x0_local, chunk_t)
        
        # Si on veut rassembler le résultat final sur le GPU 0
        if rank == 0:
            # Rassembler toutes les parties
            gathered_videos = [torch.empty_like(videos_local[0]) for _ in range(world_size)]
            dist.gather(videos_local[0], gathered_videos, dst=0)
            
            # Combiner les vidéos
            # Note: La façon de combiner dépend de comment Ulysses divise les données
            videos = self._combine_ulysses_videos(gathered_videos)
        else:
            dist.gather(videos_local[0], dst=0)
            videos = None
    else:
        # Mode single GPU
        chunk_t = int(os.environ.get("WAN_VAE_DECODE_CHUNK_T", "20"))
        videos = self._decode_in_chunks(x0, chunk_t)
    
    return videos[0] if self.rank == 0 else None


# Solution 3: Configuration recommandée pour votre cas (2x RTX 5090)
"""
Pour 2x RTX 5090 avec 32GB chacune, voici la configuration optimale:

1. Réduire WAN_VAE_DECODE_CHUNK_T:
   export WAN_VAE_DECODE_CHUNK_T=8  # ou même 6 si nécessaire

2. Augmenter la taille du pool de mémoire fragmentable:
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

3. Forcer la synchronisation entre GPU:
   export NCCL_P2P_DISABLE=1  # Désactiver P2P si problèmes de mémoire

4. Si possible, modifier le code pour utiliser le décodage CPU:
   - Décoder sur CPU avec chunks très petits
   - Puis transférer le résultat final sur GPU

5. Alternative: Utiliser gradient checkpointing pour le VAE
"""

# Solution 4: Patch minimal pour tester
def apply_minimal_fix():
    """
    Patch minimal à appliquer dans wan/textimage2video.py
    Remplacer les lignes 420-422
    """
    old_code = """
            if self.rank == 0:
                chunk_t = int(os.environ.get("WAN_VAE_DECODE_CHUNK_T", "20"))
                videos = self._decode_in_chunks(x0, chunk_t)
    """
    
    new_code = """
            if self.rank == 0:
                # Réduire chunk_t pour Ulysses multi-GPU
                default_chunk = "8" if dist.is_initialized() and dist.get_world_size() > 1 else "20"
                chunk_t = int(os.environ.get("WAN_VAE_DECODE_CHUNK_T", default_chunk))
                
                # Forcer le modèle sur CPU avant décodage pour libérer de la VRAM
                if offload_model and hasattr(self, 'model'):
                    self.model.cpu()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                videos = self._decode_in_chunks(x0, chunk_t)
    """
    
    return old_code, new_code

if __name__ == "__main__":
    print("Solutions pour l'erreur OOM avec Ulysses:")
    print("\n1. Configuration immédiate (sans modification de code):")
    print("   export WAN_VAE_DECODE_CHUNK_T=6")
    print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128")
    print("\n2. Pour appliquer le patch minimal:")
    old, new = apply_minimal_fix()
    print(f"   Remplacer dans wan/textimage2video.py:")
    print(f"   {old}")
    print(f"   Par:")
    print(f"   {new}")