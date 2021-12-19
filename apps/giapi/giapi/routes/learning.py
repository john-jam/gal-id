import numpy as np
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from beanie import PydanticObjectId
from gicommon.routes import LEARNING_PREFIX, DL_MODELS_PATH, BASE_MODEL_NAMES_PATH, EVALUATION_PATH, PREDICTION_PATH
from gicommon.models.learning import DLModel, DLModelInDB, DLModelIn, DLModelOut, KerasModelEnum, Evaluation, \
    EvaluationIn, PredictionIn, PredictionOut
from gicommon.models.preprocessing import Dataset, DatasetInDB
from giapi.learning.model import DLModelManager
from giapi.learning.runner import TrainRunner, EvaluationRunner, PredictionRunner
from giapi.logging import logger

router = APIRouter(prefix=LEARNING_PREFIX)


@router.post(DL_MODELS_PATH, response_model=DLModelOut, status_code=status.HTTP_201_CREATED)
async def post_dl_model(dl_model_in: DLModelIn, background_tasks: BackgroundTasks):
    await must_be_available()

    dataset_in_db = await DatasetInDB.get(dl_model_in.dataset_id)
    if dataset_in_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Dataset with id {dl_model_in.dataset_id} not found")
    if not dataset_in_db.split:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Dataset with id {dl_model_in.dataset_id} not ready yet")
    dataset = Dataset.parse_obj(dataset_in_db)

    dl_model_in_db = DLModelInDB(
        **dl_model_in.dict(),
        dataset=dataset,
        busy=True,
        trained=False,
    )
    await dl_model_in_db.create()
    dl_model = DLModel.parse_obj(dl_model_in_db)

    background_tasks.add_task(train_background_task, dl_model)
    background_tasks.add_task(after_train_background_task, dl_model_in_db)

    return dl_model


@router.get(DL_MODELS_PATH, response_model=List[DLModelOut])
async def get_all_dl_models():
    dl_models_in_db = await DLModelInDB.find_all().to_list()
    return [DLModel.parse_obj(x) for x in dl_models_in_db]


@router.post(EVALUATION_PATH.format('{dl_model_id}'), response_model=DLModelOut)
async def post_evaluation(dl_model_id: PydanticObjectId, evaluation: EvaluationIn):
    await must_be_available()

    dl_model_in_db = await DLModelInDB.get(dl_model_id)
    if dl_model_in_db is None:
        raise HTTPException(status_code=404, detail=f"Deep Learning Model with id {dl_model_id} not found")
    dl_model = DLModel.parse_obj(dl_model_in_db)

    await update_dl_model_status(dl_model_in_db, busy=True)
    train_scores, test_scores = EvaluationRunner(DLModelManager(dl_model)).evaluate(evaluation.batch_size)
    await update_dl_model_status(dl_model_in_db, busy=False)

    dl_model_in_db.evaluation = Evaluation(
        train_loss=train_scores[0],
        train_accuracy=train_scores[1],
        test_loss=test_scores[0],
        test_accuracy=test_scores[1],
    )
    await dl_model_in_db.save()
    dl_model = DLModel.parse_obj(dl_model_in_db)

    return dl_model


@router.post(PREDICTION_PATH.format('{dl_model_id}'), response_model=PredictionOut)
async def post_prediction(dl_model_id: PydanticObjectId, prediction: PredictionIn):
    await must_be_available()

    dl_model_in_db = await DLModelInDB.get(dl_model_id)
    if dl_model_in_db is None:
        raise HTTPException(status_code=404, detail=f"Deep Learning Model with id {dl_model_id} not found")
    dl_model = DLModel.parse_obj(dl_model_in_db)

    image = np.array(prediction.image, dtype='uint8')

    await update_dl_model_status(dl_model_in_db, busy=True)
    probabilities, category = PredictionRunner(DLModelManager(dl_model)).predict(image)
    await update_dl_model_status(dl_model_in_db, busy=False)
    return PredictionOut(
        probabilities=probabilities.tolist(),
        category=category
    )


@router.get(BASE_MODEL_NAMES_PATH, response_model=List[str])
async def get_all_base_model_names():
    return KerasModelEnum.to_list()


# TODO: remove
@router.get('/debug-clear')
async def debug_clear():
    await DLModelInDB.find_many({'busy': True}).update({"$set": {'busy': False, 'trained': True}})
    return 'done'


def train_background_task(dl_model: DLModel):
    TrainRunner(DLModelManager(dl_model)).train()


async def after_train_background_task(dl_model_in_db: DLModelInDB):
    dl_model_in_db.trained = True
    dl_model_in_db.busy = False
    await dl_model_in_db.save()
    logger.info(f'DLModel {dl_model_in_db.id} updated')


async def update_dl_model_status(dl_model_in_db: DLModelInDB, busy):
    dl_model_in_db.busy = busy
    await dl_model_in_db.save()


async def must_be_available():
    dl_model_in_db_busy = await DLModelInDB.find_one({'busy': True})
    if dl_model_in_db_busy:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Learning platform is busy, try again later")
