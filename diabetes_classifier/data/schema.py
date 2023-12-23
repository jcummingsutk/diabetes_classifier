from enum import Enum, auto


class DataType(Enum):
    nominal = auto()
    ordinal = auto()
    discrete = auto()
    continuous = auto()


SCHEMA: dict[str, DataType] = {
    "Diabetes": DataType.nominal,
    "HighBP": DataType.nominal,
    "HighChol": DataType.nominal,
    "CholCheck": DataType.nominal,
    "BMI": DataType.discrete,
    "Smoker": DataType.nominal,
    "Stroke": DataType.nominal,
    "HeartDiseaseorAttack": DataType.nominal,
    "PhysActivity": DataType.nominal,
    "Fruits": DataType.nominal,
    "Veggies": DataType.nominal,
    "HvyAlcoholConsump": DataType.nominal,
    "AnyHealthcare": DataType.nominal,
    "NoDocbcCost": DataType.nominal,
    "GenHlth": DataType.ordinal,
    "MentHlth": DataType.discrete,
    "PhysHlth": DataType.discrete,
    "DiffWalk": DataType.nominal,
    "Sex": DataType.nominal,
    "Age": DataType.ordinal,
    "Education": DataType.ordinal,
    "Income": DataType.ordinal,
}
