# Predizione del Costo di Polizze Sanitarie tramite Modelli di Regressione

## Descrizione del Progetto
Sviluppo di una pipeline di Machine Learning in ambiente Python/Jupyter finalizzata alla stima del premio assicurativo annuale (`charges`) a partire da variabili socio-demografiche e biometriche individuali. L'analisi include la preparazione dei dati, l'analisi esplorativa, l'ingegnerizzazione delle feature e il confronto quantitativo tra modelli lineari e modelli basati su insiemi di alberi decisionali.
* **Autore:** Cristiano De Luca
* **Dataset:** Medical Cost Personal Datasets (Kaggle)

## Dataset e Variabili
Il dataset è composto da 1.338 osservazioni prive di valori nulli. Le variabili considerate sono:
* `age`: Età anagrafica dell'assicurato (18-64 anni).
* `sex`: Genere del beneficiario (`male`, `female`).
* `bmi`: Indice di Massa Corporea (`float`, media 30.66).
* `children`: Numero di figli o persone a carico nel nucleo familiare (0-5).
* `smoker`: Abitudine al fumo (`yes`, `no`).
* `region`: Zona geografica di residenza negli USA (`northeast`, `northwest`, `southeast`, `southwest`).
* **Target** `charges`: Costo annuale della polizza sanitaria espresso in valuta.

## Requisiti Tecnici
* **Linguaggio:** Python 3.10+
* **Librerie Principali:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Metodologia di Sviluppo
1. **Exploratory Data Analysis (EDA):** Ispezione della distribuzione delle variabili numeriche e studio dell'asimmetria del target (skewness originale pari a 1.516). Individuazione dei principali driver di costo mediante boxplot e scatter plot, che evidenziano una marcata correlazione tra status di fumatore (`smoker_yes`) ed elevata spesa assicurativa.
2. **Data Preprocessing:** Applicazione del One-Hot Encoding con rimozione della prima classe (`drop_first=True`) per le variabili categoriche (`sex`, `smoker`, `region`). Suddivisione del dataset in Training Set (80%, 1.070 campioni) e Test Set (20%, 268 campioni) con `random_state=42`. Standardizzazione delle feature numeriche tramite `StandardScaler` per i modelli lineari.
3. **Model Selection:** Addestramento e validazione comparativa di un modello di Regressione Lineare (su dati scalati) e di un Random Forest Regressor (100 stimatori).
4. **Feature Importance:** Calcolo dell'importanza relativa delle feature tramite Random Forest, confermando che la variabile `smoker_yes` rappresenta il fattore discriminante primario, seguita da `bmi` ed `age`.

## Risultati e Confronto Prestazioni
I modelli sono stati valutati sul test set calcolando Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) e Coefficiente di Determinazione (R²):

| Modello | MAE ($) | RMSE ($) | R² |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **2550.08** | **4576.30** | **0.8651** |
| **Linear Regression** | 4181.19 | 5796.28 | 0.7836 |

Il Random Forest si è dimostrato significativamente superiore, spiegando oltre l'86% della varianza totale e riducendo l'errore medio di previsione di oltre 1.600 dollari rispetto alla Regressione Lineare, grazie alla capacità di catturare le interazioni non lineari tra indice di massa corporea ed abitudine al fumo.

## Istruzioni di Esecuzione
1. Clonare il repository:
   ```bash
   git clone https://github.com
   ```
2. Installare i pacchetti necessari:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Avviare il notebook:
   ```bash
   jupyter notebook
   ```
