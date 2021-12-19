from pydantic import BaseModel, PositiveInt, NonNegativeInt, confloat
from gicommon.models.base import BaseInCode, BaseInDB


class DatasetBase(BaseModel):
    test_size: confloat(ge=0.01, le=0.99)
    random_state: NonNegativeInt
    image_size: PositiveInt
    compressed: bool


class DatasetBaseExtra(DatasetBase):
    split: bool


class Dataset(BaseInCode, DatasetBaseExtra):
    pass


class DatasetInDB(BaseInDB, DatasetBaseExtra):
    pass


class DatasetIn(DatasetBase):
    pass


class DatasetOut(Dataset):
    pass
