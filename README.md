# Trading Stack — NVDA Transformer

## Structure
```
trading/
├── docker-compose.yml
├── .env                        ← tes identifiants IB
├── models/
│   ├── nvda_transformer.keras  ← généré par nvda_transformer.py
│   └── scaler.pkl              ← généré par nvda_transformer.py
├── model-api/
│   ├── Dockerfile
│   ├── api_model.py
│   └── requirements.txt
└── bot/
    ├── Dockerfile
    ├── bot.py
    └── requirements.txt
```

## Prérequis

1. Avoir entraîné le modèle et récupéré les fichiers :
   - `nvda_transformer.keras`
   - `scaler.pkl`

   Copie-les dans le dossier `models/` :
   ```bash
   cp nvda_transformer.keras trading/models/
   cp scaler.pkl             trading/models/
   ```

2. Renseigner le fichier `.env` avec tes identifiants IB.

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
