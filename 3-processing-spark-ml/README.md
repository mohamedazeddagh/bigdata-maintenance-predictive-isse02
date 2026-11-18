\# Module 3 : Traitement Distribué \& Machine Learning (Apache Spark)



Ce module implémente le pipeline de \*\*Feature Engineering (Spark SQL)\*\* et de \*\*Prédiction de la Durée de Vie Résiduelle — RUL (Spark MLlib)\*\* sur le jeu de données NASA C-MAPSS (`FD001`).



\## Architecture du Module



1\. \*\*`train\_rul\_model.py` (Entraînement \& Évaluation)\*\* :

&#x20;  - Chargement et nettoyage des 21 capteurs et 3 paramètres opérationnels.

&#x20;  - Calcul de la variable cible `RUL` avec plafonnement industriel à \*\*125 cycles\*\* (\*RUL Clipping\*).

&#x20;  - Extraction de variables temporelles glissantes sur une fenêtre de \*\*5 cycles\*\* (`ROWS BETWEEN 4 PRECEDING AND CURRENT ROW`) via \*\*Spark SQL\*\* sur les 14 capteurs pertinents.

&#x20;  - Entraînement d'un modèle \*\*GBTRegressor\*\* (\*Gradient Boosted Trees\*, `maxIter=50`, `maxDepth=5`).

&#x20;  - \*\*Performance obtenue :\*\* \*\*RMSE = 17.35 cycles\*\* sur le jeu de test (20%).



2\. \*\*`predict\_rul\_batch.py` (Inférence Batch \& Couche Gold)\*\* :

&#x20;  - Application du modèle entraîné sur les nouvelles mesures capteurs (`test\_FD001.txt` / Zone Silver).

&#x20;  - Classification de l'état de santé actuel de chaque moteur pour la couche \*\*Gold\*\* et l'alerte \*\*Grafana\*\* :

&#x20;    - \*\*Critique\*\* : `RUL <= 20` cycles

&#x20;    - \*\*À surveiller\*\* : `20 < RUL <= 50` cycles

&#x20;    - \*\*Normal\*\* : `RUL > 50` cycles



\## Exécution Locale



```bash

python 3-processing-spark-ml/train\_rul\_model.py

python 3-processing-spark-ml/predict\_rul\_batch.py

