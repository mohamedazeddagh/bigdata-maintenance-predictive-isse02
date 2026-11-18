import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline


def compute_rolling_features(spark, df, capteurs_utiles):
    """Applique la fenêtre glissante Spark SQL sur 5 cycles."""
    df.createOrReplaceTempView("silver_capteurs")
    rolling_exprs = [
        f"AVG({c}) OVER (PARTITION BY id_moteur ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS {c}_moy_5c"
        for c in capteurs_utiles
    ]
    sql_query = f"SELECT *, {', '.join(rolling_exprs)} FROM silver_capteurs"
    return spark.sql(sql_query)


def load_cmapss_file(spark, path):
    """Charge et nettoie un fichier NASA C-MAPSS (train ou test)."""
    raw_df = spark.read.csv(path, sep=" ", inferSchema=True)
    df = raw_df.select(*raw_df.columns[:26])
    col_names = ["id_moteur", "cycle", "setting_1", "setting_2", "setting_3"] + [
        f"s{i}" for i in range(1, 22)
    ]
    return df.toDF(*col_names)


def main():
    spark = (
        SparkSession.builder.appName("ISSE02-Batch-Inference-Gold")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(">>> Session Spark (Inférence Gold) démarrée.")

    capteurs_utiles = [
        "s2", "s3", "s4", "s7", "s8", "s9", "s11",
        "s12", "s13", "s14", "s15", "s17", "s20", "s21"
    ]
    feature_cols = capteurs_utiles + [f"{c}_moy_5c" for c in capteurs_utiles]
    model_path = "3-processing-spark-ml/gbt_rul_model"

    # 1. Chargement du modèle sauvegardé (ou ré-entraînement rapide si absent/Windows)
    model = None
    if os.path.exists(model_path):
        try:
            print(f">>> Chargement du modèle depuis '{model_path}'...")
            model = PipelineModel.load(model_path)
        except Exception:
            print(">>> Modèle local invalide sous Windows : bascule vers l'entraînement rapide...")

    if model is None:
        print(">>> Entraînement rapide du modèle sur train_FD001.txt...")
        train_df = load_cmapss_file(spark, "data/train_FD001.txt")
        max_cycles = train_df.groupBy("id_moteur").agg(F.max("cycle").alias("max_cycle"))
        train_df = (
            train_df.join(max_cycles, on="id_moteur", how="inner")
            .withColumn("RUL", F.col("max_cycle") - F.col("cycle"))
            .drop("max_cycle")
            .withColumn("RUL", F.when(F.col("RUL") > 125, 125).otherwise(F.col("RUL")))
        )
        train_features = compute_rolling_features(spark, train_df, capteurs_utiles)
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        gbt = GBTRegressor(featuresCol="features", labelCol="RUL", maxIter=50, maxDepth=5, seed=42)
        model = Pipeline(stages=[assembler, gbt]).fit(train_features)

    # 2. Lecture des données de test (Simulation de la Zone Silver)
    test_path = "data/test_FD001.txt" if os.path.exists("data/test_FD001.txt") else "data/train_FD001.txt"
    print(f">>> Lecture des données capteurs depuis '{test_path}'...")
    silver_df = load_cmapss_file(spark, test_path)

    # 3. Calcul des variables glissantes
    silver_features = compute_rolling_features(spark, silver_df, capteurs_utiles)

    # 4. Prédiction RUL et génération de la table Gold
    predictions = model.transform(silver_features)

    # On garde le dernier cycle connu pour chaque moteur (état actuel de la flotte pour Grafana)
    latest_cycles = predictions.groupBy("id_moteur").agg(F.max("cycle").alias("dernier_cycle"))
    gold_fleet_df = (
        predictions.join(
            latest_cycles,
            (predictions.id_moteur == latest_cycles.id_moteur) & (predictions.cycle == latest_cycles.dernier_cycle),
            "inner"
        )
        .drop(latest_cycles.id_moteur)
        .drop("dernier_cycle")
        .withColumn("RUL_predite", F.round("prediction", 1))
        .withColumn(
            "etat_sante",
            F.when(F.col("RUL_predite") <= 20, "Critique")
            .when(F.col("RUL_predite") <= 50, "A surveiller")
            .otherwise("Normal"),
        )
        .withColumn("timestamp_calcul", F.current_timestamp())
        .select("id_moteur", "cycle", "RUL_predite", "etat_sante", "timestamp_calcul")
        .orderBy("RUL_predite")
    )

    print("\n>>> État de santé actuel de la flotte (Top 15 moteurs les plus critiques — Zone Gold) :")
    gold_fleet_df.show(15, truncate=False)

    # 5. Export de la table Gold en CSV via Pandas
    os.makedirs("data/gold_output", exist_ok=True)
    gold_pdf = gold_fleet_df.toPandas()
    output_csv = "data/gold_output/etat_sante_moteurs.csv"
    gold_pdf.to_csv(output_csv, index=False)
    print(f">>> Table Gold exportée avec succès dans '{output_csv}' ({len(gold_pdf)} moteurs analysés).")

    spark.stop()


if __name__ == "__main__":
    main()