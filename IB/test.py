from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta

# Connexion à TWS (ou IB Gateway)
ib = IB()
ib.connect("127.0.0.1", 7497, clientId=1)  # 7497 = TWS / 4001 = Gateway

# Définir le contrat NVIDIA
contract = Stock("NVDA", "SMART", "USD")

# Liste pour stocker les données
all_data = []

# Fenêtre de 1 jour à chaque itération (granularité 10 mins max sur 1 journée)
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 ans en arrière

current = end_date

# Itérer sur chaque jour en arrière
while current > start_date:
    print(f"Récupération: {current.strftime('%Y-%m-%d')}")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=current,
        durationStr="1 D",
        barSizeSetting="10 mins",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
    )

    if bars:
        df = util.df(bars)
        all_data.append(df)

    # Avancer d'un jour
    current -= timedelta(days=1)
    ib.sleep(2)  # Respecter le rate limit

# Fusionner les données
df_all = pd.concat(all_data)
df_all.drop_duplicates(inplace=True)

# Sauvegarder
df_all.to_csv("NVDA_10min_2years.csv", index=False)
print("Données sauvegardées dans NVDA_10min_2years.csv")

ib.disconnect()
