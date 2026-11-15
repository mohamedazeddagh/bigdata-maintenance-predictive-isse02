from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def main():
    # 1. Initialisation de la session Spark
    spark = SparkSession.builder \
        .appName("ISSE02-Predictive-Maintenance-RUL") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print(">>> Session Spark démarrée avec succès.")

    # 2. Chargement du dataset NASA C-MAPSS (train_FD001.txt)
    data_path = "data/train_FD001.txt"
    raw_df = spark.read.csv(data_path, sep=" ", inferSchema=True)

    # Le fichier contient 2 colonnes vides à la fin à cause des espaces doubles
    valid_cols = raw_df.columns[:26]
    df = raw_df.select(*valid_cols)

    # Attribution des noms de colonnes officiels
    col_names = ["id_moteur", "cycle", "setting_1", "setting_2", "setting_3"] + \
                [f"s{i}" for i in range(1, 22)]
    df = df.toDF(*col_names)

    # 3. Calcul de la variable cible : RUL (Remaining Useful Life)
    max_cycles = df.groupBy("id_moteur").agg(F.max("cycle").alias("max_cycle"))
    df = df.join(max_cycles, on="id_moteur", how="inner") \
           .withColumn("RUL", F.col("max_cycle") - F.col("cycle")) \
           .drop("max_cycle")

    # Plafonnement (RUL clipping à 125 cycles, standard industriel pour C-MAPSS)
    df = df.withColumn("RUL", F.when(F.col("RUL") > 125, 125).otherwise(F.col("RUL")))

    # 4. Feature Engineering avec Spark SQL : Variables glissantes (fenêtre de 5 cycles)
    df.createOrReplaceTempView("capteurs_view")

    # Sélection des capteurs pertinents (ceux qui varient réellement dans FD001)
    capteurs_utiles = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    
    rolling_exprs = [
        f"AVG({c}) OVER (PARTITION BY id_moteur ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS {c}_moy_5c"
        for c in capteurs_utiles
    ]
    
    sql_query = f"""
        SELECT *, {', '.join(rolling_exprs)}
        FROM capteurs_view
    """
    df_features = spark.sql(sql_query)
    print(">>> Variables glissantes calculées via Spark SQL.")

    # 5. Préparation des données pour Spark MLlib
    feature_cols = capteurs_utiles + [f"{c}_moy_5c" for c in capteurs_utiles]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Séparation Train (80%) / Test (20%)
    train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

    # Modèle GBTRegressor (Gradient Boosted Trees)
    gbt = GBTRegressor(featuresCol="features", labelCol="RUL", maxIter=50, maxDepth=5, seed=42)
    pipeline = Pipeline(stages=[assembler, gbt])

    print(">>> Entraînement du modèle GBTRegressor en cours...")
    model = pipeline.fit(train_df)

    # 6. Évaluation du modèle (RMSE)
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(labelCol="RUL", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f">>> Performance du modèle — RMSE sur le jeu de test : {rmse:.2f} cycles")

    # 7. Classification de l'état de santé (prêt pour la zone Gold & Grafana)
    gold_df = predictions.withColumn(
        "etat_sante",
        F.when(F.col("prediction") <= 20, "Critique")
         .when(F.col("prediction") <= 50, "A surveiller")
         .otherwise("Normal")
    )

    print("\n>>> Aperçu des résultats (Zone Gold) :")
    gold_df.select("id_moteur", "cycle", "RUL", F.round("prediction", 1).alias("RUL_predite"), "etat_sante") \
           .show(15, truncate=False)

    # 8. Sauvegarde du modèle entraîné (compatible Linux/Docker et Windows)
    try:
        model.write().overwrite().save("3-processing-spark-ml/gbt_rul_model")
        print(">>> Modèle sauvegardé dans '3-processing-spark-ml/gbt_rul_model'.")
    except Exception:
        print(">>> Note : Sauvegarde Hadoop locale ignorée sous Windows (fonctionnera nativement sous Docker/Linux).")

    spark.stop()

if __name__ == "__main__":
    main()