# This script will create the "original" data, although still synthetic in nature.
import polars as pl 
import uuid
import numpy as np

def generate_data(num_patients: int = 100, random_seed: int|None = None, include_patient_id: bool = False) -> pl.DataFrame:
    """
    Generates a synthetic dataset using predefined distributions. The following elements are synthetically generated: patient age, patient sex, patient race,
    height, weight, diastolic blood pressure, systolic blood pressure, and heart rate.

    Parameters:

        num_patients (int): Number of synthetic patients (rows) to be generated.

        random_seed (int): Seed value for reproducibility. 

        include_patient_id (bool): True if a unique UUID4 patient identifier is desired in final output, otherwise False.

    Returns:

        polars.DataFrame object
    """
    np.random.seed(random_seed)
    data = []
    for _ in range(num_patients):
        # set patient id, which can later be excluded from output if include_patient_id = False
        pid = uuid.uuid4()

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

        row = [str(pid), int(age[0]), str(sex[0]), str(race[0]), int(height), int(weight), int(systolic_bp[0]), int(diastolic_bp[0]), int(heart_rate[0])]
        data.append(row)
    df = pl.DataFrame(
        data, 
        orient='row',
        schema = {
            "patient_id":pl.Utf8,
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

    if not include_patient_id:
        return df.drop("patient_id")
    else:
        return df
    
if __name__ == "__main__":
    data = generate_data(num_patients=100000, random_seed=42, include_patient_id=True)
    data.write_parquet('data/original_data.parquet')