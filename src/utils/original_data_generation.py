# This script will create the "original" data, although still synthetic in nature.
import polars as pl 
import uuid
import numpy as np


def generate_data(num_patients, random_seed):
    np.random.seed(random_seed)
    data = []
    for _ in range(num_patients):
        # set age 
        age = np.random.randint(18, 90, size = 1)

        # random selection with probabilities
        sex = np.random.choice(
            ["Male","Female","Other","Unknown"], 
            size = 1, 
            p = [0.45, 0.48, 0.02, 0.05]
        )

        # random selection with probabilities
        race = np.random.choice(
            ["White","Black or African American","Asian","American Indian or Alaska Native","Native Hawaiian or Pacific Islander","Other Race","Unknown"], 
            size = 1, 
            p = [0.67, 0.16, 0.08, 0.02, 0.02, 0.01, 0.04]
        )

        # conditional based on gender
        height = np.random.normal(
            loc = 167 + (8 if sex == "Male" else (-5 if sex == "Female" else 0)),
            scale = 6.5 + (0.5 if sex == "Male" else (-0.5 if sex == "Female" else 0))
        )

        # first calculate bmi to derive weight 
        bmi = np.random.normal(
            loc = 27 + (0 if race == "Male" else -0.5) + (0 if age < 50 else 1),
            scale = 5
        )
        weight = bmi * (height / 100.0) ** 2

        # calculate blood pressure readings + heart rate - dependent on age & partially to each other
        systolic_bp = 110 + 0.5*(age - 40) + np.random.normal(loc = 0, scale = 12, size = 1)
        diastolic_bp = 0.6*systolic_bp + np.random.normal(loc = 0, scale = 8, size = 1)
        heart_rate = 75 - 0.05*(systolic_bp-120) + np.random.normal(loc = 0, scale = 7, size = 1)

        # clip blood pressures and HR to realistic values
        systolic_bp = np.clip(systolic_bp, 90, 200)
        diastolic_bp = np.clip(diastolic_bp, 50, 120)
        heart_rate = np.clip(heart_rate, 40, 140)

        row = [int(age[0]), str(sex[0]), str(race[0]), int(height), int(weight), int(systolic_bp[0]), int(diastolic_bp[0]), int(heart_rate[0])]
        data.append(row)
    df = pl.DataFrame(
        data, 
        orient='row',
        schema = {
            "patient_age":pl.Int32, 
            "patient_gender":pl.Utf8, 
            "patient_race":pl.Utf8, 
            "patient_height_cm":pl.Int32, 
            "patient_weight_kg":pl.Int32, 
            "patient_systolic_bp":pl.Int32, 
            "patient_diastolic_bp":pl.Int32, 
            "patient_heart_rate":pl.Int32
        }
    )
    df = df.with_columns(
        pl.when(pl.col("patient_gender") == "Unknown").then(None).otherwise(pl.col("patient_gender")).alias("patient_gender"),
        pl.when(pl.col("patient_race") == "Unknown").then(None).otherwise(pl.col("patient_race")).alias("patient_race"),
        pl.when(pl.lit(np.random.rand(df.height)) < 0.05).then(None).otherwise(pl.col("patient_diastolic_bp")).alias("patient_diastolic_bp"),
        pl.when(pl.lit(np.random.rand(df.height)) < 0.05).then(None).otherwise(pl.col("patient_systolic_bp")).alias("patient_systolic_bp"),
        pl.when(pl.lit(np.random.rand(df.height)) < 0.05).then(None).otherwise(pl.col("patient_heart_rate")).alias("patient_heart_rate")
    )

    return df