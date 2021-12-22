from typing import List
from fastapi import APIRouter, BackgroundTasks, status
from gicommon.routes import PREPROCESSING_PREFIX, DATASETS_PATH
from gicommon.models.preprocessing import Dataset, DatasetInDB, DatasetIn, DatasetOut
from giapi.preprocessing.dataset import DatasetManager
from giapi.logging import logger

router = APIRouter(prefix=PREPROCESSING_PREFIX)


@router.post(DATASETS_PATH, response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def create_dataset(dataset_in: DatasetIn, background_tasks: BackgroundTasks):
    dataset_in_db = DatasetInDB(
        **dataset_in.dict(),
        split=False
    )
    await dataset_in_db.create()
    dataset = Dataset.parse_obj(dataset_in_db)

    background_tasks.add_task(split_task, dataset)
    background_tasks.add_task(after_split_task, dataset_in_db)

    return dataset


@router.get(DATASETS_PATH, response_model=List[DatasetOut])
async def get_datasets():
    datasets_in_db = await DatasetInDB.find_all().to_list()
    return [Dataset.parse_obj(x) for x in datasets_in_db]


def split_task(dataset: Dataset):
    manager = DatasetManager(dataset)
    manager.split()


async def after_split_task(dataset_in_db: DatasetInDB):
    dataset_in_db.split = True
    await dataset_in_db.save()
    logger.info(f'Dataset {dataset_in_db.id} updated')
