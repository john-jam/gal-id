import numpy as np
from multiprocessing import Process, Queue
from giapi.learning.model import DLModelManager


class TrainRunner:
    def __init__(self, dl_manager: DLModelManager):
        self._dl_manager = dl_manager

    def train(self):
        process = Process(target=self._dl_manager.train)
        process.start()
        process.join()


class EvaluationRunner:
    def __init__(self, dl_manager: DLModelManager):
        self.__dl_manager = dl_manager
        self.__queue = Queue()

    def _evaluate(self, batch_size):
        train_scores, test_scores = self.__dl_manager.evaluate(batch_size)
        self.__queue.put([train_scores, test_scores])

    def evaluate(self, batch_size):
        process = Process(target=self._evaluate, args=(batch_size,))
        process.start()
        process.join()
        train_scores, test_scores = self.__queue.get_nowait()
        return train_scores, test_scores


class PredictionRunner:
    def __init__(self, dl_manager: DLModelManager):
        self.__dl_manager = dl_manager
        self.__queue = Queue()

    def _predict(self, image: np.ndarray):
        probabilities, category = self.__dl_manager.predict(image)
        self.__queue.put([probabilities, category])

    def predict(self, image: np.ndarray):
        process = Process(target=self._predict, args=(image,))
        process.start()
        process.join()
        probabilities, category = self.__queue.get_nowait()
        return probabilities, category
