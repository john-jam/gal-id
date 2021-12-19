from gicommon.models.preprocessing import DatasetOut
from gicommon.models.learning import DLModelOut


def get_item_from_option(items, option):
    return next((x for x in items if x.id == option.id), None)


def format_dl_model_option(dl_model: DLModelOut):
    training_prefix = "(training) " if not dl_model.trained else ""
    return f'{training_prefix}Base model: {dl_model.base_model_name} | Epochs: {dl_model.epochs} | Image size: {dl_model.dataset.image_size} | {dl_model.created_at.strftime("%Y-%m-%d %H:%M:%S")}'


def get_dl_models_options(dl_models, trained_only=False):
    if trained_only:
        dl_models = [dl_model for dl_model in dl_models if dl_model.trained]
    return [
        DLModelOut.construct(
            id=dl_model.id,
            base_model_name=dl_model.base_model_name,
            epochs=dl_model.epochs,
            dataset=dl_model.dataset,
            created_at=dl_model.created_at,
            trained=dl_model.trained
        )
        for dl_model in dl_models
    ]


def get_dataset_options(datasets, split_only=False):
    if split_only:
        datasets = [dataset for dataset in datasets if dataset.split]
    return [
        DatasetOut.construct(
            id=dataset.id,
            image_size=dataset.image_size,
            compressed=dataset.compressed,
            created_at=dataset.created_at,
            split=dataset.split
        )
        for dataset in datasets
    ]


def format_dataset_option(dataset: DatasetOut):
    splitting_prefix = "(splitting) " if not dataset.split else ""
    return f'{splitting_prefix}Image size: {dataset.image_size} | Compressed: {dataset.compressed} | {dataset.created_at.strftime("%Y-%m-%d %H:%M:%S")}'
