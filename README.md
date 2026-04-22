# Financial Bot — NVDA Transformer

## Structure
```
financial-bot/
├── docker-compose.yml
├── .env                        ← credentials IB (host, port, client_id)
├── models/
│   ├── nvda_transformer.keras  ← entraîné par model_training
│   └── scaler.pkl              ← fit sur les 32 dernières barres
├── model-api/                 ← Flask API qui expose /predict & /health
│   ├── Dockerfile
│   ├── api_model.py
│   └── requirements.txt
├── bot/                       ← trading loop (paper trading par défaut)
│   ├── bot.py
│   └── requirements.txt
└── data/                      ← historique OHLCV (optionnel, ex: NVDA_10min_2years.csv)
```

## Prérequis

1. Avoir entraîné le modèle et récupéré les fichiers :
   - `nvda_transformer.keras`
   - `scaler.pkl`

   Copie-les dans le dossier `models/` :
   ```bash
   cp models/nvda_transformer.keras financial-bot/models/
   cp models/scaler.pkl               financial-bot/models/
   ```

2. Renseigner le fichier `.env` avec les credentials IB :
   ```env
   IB_HOST=ib-gateway
   IB_PORT=4002           # paper trading (4001 pour live)
   IB_CLIENT_ID=10
   MODEL_API_URL=http://model-api:5000
   SYMBOL=NVDA
   ```

## Démarrage

```bash
# Premier lancement (build des images)
docker compose up --build

# Lancement en arrière-plan
docker compose up -d

# Logs du bot
docker compose logs -f bot

# Logs de l'API modèle
docker compose logs -f model-api
```

## Tester l'API modèle manuellement

```bash
# Health check
curl http://localhost:5000/health

# Prédiction (7 features × 32 barres = 224 valeurs)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bars": [
      [130.0, 131.5, 129.8, 130.9, 150000, 130.5, 420],
      ... (32 barres totales)
    ]
  }'
```

## Mettre à jour le modèle

Après réentraînement, copie les nouveaux fichiers dans `models/` et redémarre uniquement le service model-api :

```bash
docker compose restart model-api
```

Le bot continue de tourner sans interruption.

## Paper vs Live Trading

Par défaut, le bot est en **paper trading** (ordres simulés). Les signaux sont loggués mais non exécutés.

### Passer au live trading :

1. Dans `docker-compose.yml` ou `.env`, changer :
   ```yaml
   TRADING_MODE: live      # optionnel si déjà via .env
   IB_PORT: 4001           # switch de paper→live
   ```

2. Dans `bot/bot.py`, décommenter les lignes d'exécution des ordres (section `handle_signal`) :
   ```python
   if signal == "BUY":
       order = MarketOrder("BUY", 1)  # <--- décommenter
       trade = ib.placeOrder(contract, order)
   ```

## Comment ça marche

1. **Dataflow :**
   - Le bot demande 32 barres de 10 min (6h d'historique) à Interactive Brokers
   - Chaque barre a 7 features : Open, High, Low, Close, Volume, OI, IV

2. **Prédiction (model-api) :**
   - L'API Flask enveloppe le modèle Transformer
   - Format attendu : `{"bars": [[o,h,l,c,v,oi,iv], ...]}` (32 × 7 = 224 valeurs)
   - Retour : `"signal": "BUY" | "SELL" | "HOLD"`, prix actuel et prédit

3. **Action (bot) :**
   - Si signal BUY → acheter (optionnel en paper trading)
   - Si signal SELL → vendre toutes les positions (optionnel)
   - Sinon → rester neutre

## Entraînement du modèle

Optionnel pour commencer, mais recommandé :

```bash
# Activer venv & pip install requirements.txt
python model_training/nvda_transformer.py
# Génère models/nvda_transformer.keras et models/scaler.pkl
```

Copie ensuite les fichiers dans `financial-bot/models/` (voir plus haut).

## Données historiques

Le bot peut aussi s'entraîner sur des données locales si tu n'as pas accès au marché en direct :

- Copie un fichier `.csv` (ex: `NVDA_10min_2years.csv`) dans `financial-bot/data/`
- Lance l'entraînement via `python -m model_training.nvda_transformer --data_file data/NVDA_10min_2years.csv`

## Démarrage

```bash
# Premier lancement (build des images)
docker compose up --build

# Lancement normal
docker compose up -d

# Voir les logs du bot
docker compose logs -f bot

# Voir les logs de l'API modèle
docker compose logs -f model-api
```

## Tester l'API modèle manuellement

```bash
# Health check
curl http://localhost:5000/health

# Prédiction (remplace les valeurs par de vraies barres)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bars": [
      [130.0, 131.5, 129.8, 130.9, 150000, 130.5, 420],
      ...
    ]
  }'
```

## Mettre à jour le modèle

Après réentraînement, copie les nouveaux fichiers dans `models/` et redémarre
uniquement le service model-api :

```bash
cp nvda_transformer.keras trading/models/
cp scaler.pkl             trading/models/
docker compose restart model-api
```

Le bot continue de tourner sans interruption.

## Passer en live trading

Dans `docker-compose.yml`, changer :
```yaml
TRADING_MODE: live
IB_PORT: 4001
```

Et dans `bot/bot.py`, décommenter les lignes `MarketOrder`.
