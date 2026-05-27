# Industrial Motor Predictive Maintenance 

## Project Overview
This project applies Machine Learning to industrial telemetry data to predict equipment failure before it happens. By analyzing high-frequency sensor data (temperature, load, and vibration), the goal is to identify anomalous physical states and classify whether a machine is at risk of an imminent breakdown.

## The Engineering Problem
In industrial automation, unexpected motor failures lead to massive downtime costs. Raw vibration data is extremely noisy. A healthy motor exhibits baseline vibrations, but a failing component will cause sharp, rapid fluctuations in the signal's variance. 

## Methodology & Feature Engineering
Instead of feeding raw, noisy sensor data directly into a model, I utilized **Pandas** to engineer time-series features:
* **Rolling Window Statistics:** Applied a rolling standard deviation to the horizontal and vertical vibration signals. This acts as a digital filter, smoothing out normal noise and isolating the physical shocks/impacts that precede a mechanical failure.
* **Threshold Analysis:** Filtered and analyzed extreme states (e.g., Load > 85% and Motor Temp > 80°C) to mathematically prove a ~35% relative increase in failure probability under high-stress conditions.

## Tech Stack
* **Python**
* **Pandas & NumPy:** For data manipulation, cleaning, and time-series feature extraction.
* **Scikit-Learn:** For Train/Test splitting and model training.
* **Matplotlib & Seaborn:** For data visualization and evaluating model metrics.

## Model Training & Results
Trained an ensemble **Random Forest Classifier** to predict the `breakdown_flag`. 
* The model was evaluated using a 80/20 Train-Test split to prevent data leakage.
* Analyzed the Confusion Matrix to balance Precision and Recall, prioritizing the minimization of False Negatives (uncaught machine failures).

## How to Run
1. Clone the repository.
2. Ensure you have `pandas`, `numpy`, `scikit-learn`, and `matplotlib` installed.
3. Run the Jupyter Notebook to view the data pipeline and model evaluation.
