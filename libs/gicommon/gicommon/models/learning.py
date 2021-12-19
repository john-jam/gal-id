from typing import List, Any, Optional
from enum import Enum
from beanie import PydanticObjectId
from pydantic import BaseModel, PositiveInt, PositiveFloat
from gicommon.models.base import BaseInCode, BaseInDB
from gicommon.models.preprocessing import Dataset


class KerasModelEnum(str, Enum):
    Xception = 'Xception'
    ResNet50 = 'ResNet50'
    ResNet50V2 = 'ResNet50V2'
    InceptionV3 = 'InceptionV3'
    MobileNet = 'MobileNet'
    MobileNetV2 = 'MobileNetV2'
    DenseNet121 = 'DenseNet121'
    NASNetMobile = 'NASNetMobile'
    EfficientNetB4 = 'EfficientNetB4'

    @staticmethod
    def to_list():
        return list(map(lambda x: x.value, KerasModelEnum))


class Evaluation(BaseModel):
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float


class EvaluationIn(BaseModel):
    batch_size: PositiveInt


class DLModelBase(BaseModel):
    batch_size: PositiveInt
    base_model_name: KerasModelEnum
    use_imagenet_weights: bool
    base_model_trainable: bool
    include_relu_dense: bool
    relu_dense_units: PositiveInt
    base_learning_rate: PositiveFloat
    reduce_learning_rate_patience: PositiveInt
    epochs: PositiveInt


class DLModelBaseExtra(DLModelBase):
    dataset: Dataset
    trained: bool
    evaluation: Optional[Evaluation] = None


class DLModel(BaseInCode, DLModelBaseExtra):
    pass


class DLModelInDB(BaseInDB, DLModelBaseExtra):
    busy: bool


class DLModelIn(DLModelBase):
    dataset_id: PydanticObjectId


class DLModelOut(DLModel):
    pass


class PredictionIn(BaseModel):
    image: List[Any]


class PredictionOut(BaseModel):
    probabilities: List[float]
    category: int
