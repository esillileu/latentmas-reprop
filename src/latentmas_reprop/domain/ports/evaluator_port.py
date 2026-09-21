from abc import ABC, abstractmethod


class EvaluatorPort(ABC):
    """Port interface for domain task evaluation and answer extraction."""

    @abstractmethod
    def evaluate(
        self,
        task: str,
        raw_prediction: str,
        gold: str | None,
    ) -> tuple[str | None, bool, str | None]:
        """Evaluate raw output against ground truth.

        Returns (prediction, is_correct, error_message).
        """
        raise NotImplementedError
