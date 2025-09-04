## Module Docker de génération vidéo GPU — Design fonctionnel et architecture

### 1) Résumé en une phrase
Service REST qui orchestre `generate.py` sur plusieurs GPU via une file FIFO, expose une API de génération/polling/suppression, estime l’ETA à partir des métriques historiques et/ou des logs, et gère la conversion 704↔720 en interne.

### 2) Objectifs fonctionnels
- **API de génération**: créer un job avec `prompt`, `orientation` (1280x720 ou 720x1280), nom de fichier optionnel. Retourne `id` immédiat.
- **API de statut**: polling par `id` pour savoir si prêt, ETA, progression, et lien de téléchargement si terminé.
- **API de suppression**: supprimer par `id` et/ou par `filename`, un ou plusieurs, ou tous.
- **API utilitaire (optionnelle)**: retrouver un `id` par `prompt` (dernier job correspondant).
- **Gestion multi-GPU**: distribution intelligente des jobs sur les GPUs disponibles (aujourd’hui 0 et 1, extensible à 4+), le surplus va en queue FIFO.
- **Conversion 704 ↔ 720**: l’API accepte 720p; le module génère en 704 côté modèle puis re-encode/pad pour sortir un MP4 en 720 côté service.

### 3) Choix d’architecture (conseillé)
- **Repo séparé** (recommandé): garder Wan2.2 intact, et créer un microservice wrapper indépendant.
  - Avantages: découplage, facilité de mise à jour du modèle, CI/CD propre, permissions et sécurité mieux maîtrisées.
  - Inconvénient: pour une estimation de progression fine, on pourrait avoir besoin d’un très léger patch côté Wan2.2 (logs structurés), mais ce patch reste optionnel.
- **Alternative fork Wan2.2**: intégrer l’API directement dans le fork.
  - Avantage: meilleur accès aux événements internes pour l’ETA.
  - Inconvénients: dette de maintenance, divergences d’amont, coupling fort.

Conclusion: repo séparé + (option) mini-patch dans le fork pour logs structurés si on veut une ETA plus précise.

### 4) Composants du service
- **API HTTP** (FastAPI) exposant les routes: `POST /generate`, `GET /status/{id}`, `DELETE /jobs`/`DELETE /videos`, `GET /jobs/lookup`, `GET /files/{filename}`.
- **Orchestrateur/Planificateur**: détecte les GPUs disponibles, tient un **job queue** global, et assigne aux **workers GPU**.
- **Workers GPU**: un worker par GPU; lance `generate.py` avec `CUDA_VISIBLE_DEVICES={gpu_id}`. Capte stdout/stderr vers un log par job.
- **Stockage d’état**: SQLite (recommandé) pour persister jobs, statuts, temps historiques; JSONl possible mais moins robuste.
- **Stockage fichiers**: `/data/results` pour MP4, `/data/logs` pour logs par job. Servis via l’API.

### 5) Flux de traitement
1. Client `POST /generate` → création d’un `job` (status `queued`), calcul d’une ETA initiale basée sur historique.
2. Orchestrateur choisit le GPU avec le plus petit backlog et enfile le job dans le worker correspondant.
3. Worker GPU lance `generate.py` (env `CUDA_VISIBLE_DEVICES`) et écrit les logs dans `/data/logs/{job_id}.log`.
4. À la fin, si demandé en 720p, post-traitement `ffmpeg` (pad/scale) vers `/data/results/{filename}.mp4`.
5. Mise à jour du job: `running` → `success` ou `error`. Lien de téléchargement disponible.
6. `GET /status/{id}`: retourne progression/ETA et URL.

### 6) API (proposition)
- POST `/generate`
  - Body:
    - `prompt` (string, requis)
    - `size` (enum: `1280x720` | `720x1280`, requis; mappé en interne sur 704)
    - `filename` (string, optionnel; sinon auto) 
  - Response 202:
```
{
  "id": "uuid",
  "status": "queued",
  "queue_position": 2,
  "eta_seconds": 480,
  "gpu_assigned": null
}
```
- GET `/status/{id}`
  - Response 200:
```
{
  "id": "uuid",
  "status": "queued|running|success|error|canceled",
  "progress": 0.37,
  "eta_seconds": 305,
  "gpu_id": 1,
  "filename": "result_1_gpu_2025-01-02_12-30-00.mp4",
  "file_url": "/files/result_1_gpu_2025-01-02_12-30-00.mp4",
  "error": null
}
```
- DELETE `/jobs`
  - Query: `ids=...` (CSV) ou `all=true` (danger), ou `status=queued|running|success` pour filtrer; annule les jobs en file/exécution, supprime fichiers si terminé (option `delete_files=true`).
- DELETE `/videos`
  - Query: `filenames=...` (CSV) ou `ids=...` ou `all=true`.
- GET `/jobs/lookup?prompt=...`
  - Retourne le dernier job (ou liste) pour ce prompt (utile si seul le prompt est connu).
- GET `/files/{filename}`
  - Sert le MP4 (avec headers cache). Option: `GET /videos` pour lister.

Notes: on peut ajouter `POST /webhook` pour enregistrer un callback à appeler quand le job termine, pour éviter le polling.

### 7) Planification GPU et politique de queue
- **Détection GPU**: via `nvidia-smi -L` ou lecture `NVIDIA_VISIBLE_DEVICES`. Configurable via env `GPU_IDS=0,1`.
- **Workers**: un worker par GPU, **concurrence = 1** par GPU (évite les OOM et garantit performance stable).
- **Assignation**: au `POST /generate`, choisir le GPU avec le plus faible backlog (jobs en cours + en file). Le reste va en **FIFO**.
- **Extensibilité**: passer de 2 à 4 GPUs = workers supplémentaires, aucun changement de l’API.

### 8) Estimation de temps (ETA)
Priorité: robuste sans modifier le modèle; précision améliorée si logs structurés disponibles.
- **Tier A (idéal)**: `generate.py` émet des événements structurés (étapes/frames/%) → ETA = (reste/total) * durée moyenne par étape.
- **Tier B (heuristique)**: parsing de stdout pour détecter "step", "frame", etc.
- **Tier C (fallback)**: **moyenne mobile** des durées par GPU. 
  - ETA job `queued` = `avg_duration_per_gpu * (jobs_ahead + 1) * 1.05`.
  - ETA job `running` = `max(0, avg_duration_per_gpu - elapsed) * 1.05`.
  - Stocker l’historique par GPU (SQLite) pour se stabiliser sur 10–50 derniers jobs.

### 9) Gestion de la taille 704 ↔ 720
- Le modèle ne gère que 704 en dimension courte. 
- **Entrée API**: `1280x720` ou `720x1280`.
- **Exécution modèle**: `1280x704` ou `704x1280`.
- **Sortie 720**: post-traitement `ffmpeg`:
  - Landscape: pad vertical 8px top/bottom: `pad=1280:720:0:8:black` (évite la déformation).
  - Portrait: pad horizontal 8px left/right: `pad=720:1280:8:0:black`.
  - Alternative: scale direct à 720 (légère déformation) si on préfère éviter les bandes.

### 10) Stockage, rétention, et nommage
- **Dossiers**: `/data/results` (MP4), `/data/logs` (logs), `/data/state.db` (SQLite).
- **Nommage par défaut**: `result_gpu{gpu}_{YYYY-mm-dd_HH-MM-SS}_{short_id}.mp4`.
- **Rétention**: TTL configurable (env `RETENTION_DAYS`), routes pour suppression ciblée ou purge complète.

### 11) Observabilité et SRE
- **Logs applicatifs**: JSON vers stdout + `logs/{job_id}.log` (stdout du process de génération).
- **Healthchecks**: `GET /health` (rapide), `GET /ready` (vérifie `ckpt_dir`).
- **Metrics**: `/metrics` (Prometheus): jobs_count, durations, queue_depth, per_gpu_utilization (optionnel via `nvidia-smi`).

### 12) Sécurité et robustesse
- **Validation input**: prompt non vide, size contrôlée, filename sain.
- **Quota & rate-limit**: limiter nb jobs/heure par client si multi-tenant.
- **Isolation**: un process par job; kill propre sur cancel; libération mémoire GPU garantie.
- **Reprise après crash**: au démarrage, marquer `running` → `error` ou requeue selon politique; conserver logs.

### 13) Dockerisation (schéma)
- Base: `nvidia/cuda:12.x-runtime-ubuntu22.04` + Python 3.10/3.11.
- Dépendances: requirements (FastAPI, uvicorn, ffmpeg, sqlite, etc.) + dépendances Wan2.2.
- Variables:
  - `WAN_CKPT_DIR=/models/Wan2.2-TI2V-5B`
  - `RESULTS_DIR=/data/results`, `LOGS_DIR=/data/logs`
  - `GPU_IDS=0,1` (par défaut: auto-détection)
  - `HOST_BASE_URL` (pour construire `file_url` si derrière un reverse proxy)
- Runtime: `docker run --gpus all -v ./data:/data -p 8080:8080 ...`

### 14) Stratégie fork vs repo séparé (réponse)
- **Clarté**: le besoin est clair et bien spécifié.
- **Optimalité**: repo séparé recommandé. Éventuellement un tout petit patch côté Wan2.2 pour log/progress si on veut une ETA précise, mais non bloquant.

### 15) Points ouverts / décisions à prendre
- Progression: OK avec ETA moyenne ou on investit pour logs structurés côté modèle ?
- Post-traitement: pad (sans distorsion) vs scale (légère distorsion) pour le 704 → 720.
- Persistance: SQLite suffit ou besoin d’un Redis/RabbitMQ pour scalabilité multi-nœud ?
- Rétention: politique de purge par défaut (jours ou nombre de fichiers) ?
- Sécurité: besoin d’auth (token) sur l’API ?

### 16) Prochaines étapes (implémentation future)
- Squelette FastAPI + workers GPU + queue en mémoire + SQLite.
- Routes `POST /generate`, `GET /status/{id}`, `GET /files/{filename}`, `DELETE /jobs`, `DELETE /videos`.
- Exécution `generate.py` avec `CUDA_VISIBLE_DEVICES`, capture logs, ETA moyenne mobile.
- Post-traitement `ffmpeg` 704→720 (pad par défaut), sauvegarde MP4.
- Dockerfile + compose, variables d’env, volume `/data`.

