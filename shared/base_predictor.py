import pathlib
import typing

"""
NOTE: Any changes to this file will be overwritten by the evaluators.
"""

PathType = typing.Union[str, pathlib.Path]


class BasePredictor:
    def prediction_setup(self):
        """To be implemented by the participants.

        Participants can add the steps needed to initialize their models,
        and or any other setup related things here.
        """
        raise NotImplementedError

    def predict(self, test_set_path: PathType, predictions_output_path: PathType, register_progress):
        """To be implemented by the participants.

        Participants need to consume the test set (the path of which is passed) 
        and write the final predictions as a CSV file to `predictions_output_path`.

        Args:
            test_set_path: Path to the Test Set for the specific task.
            predictions_output_path: Output Path to write the predictions as a CSV file. 
            register_progress: A helper callable to register progress. Accepts a value [0, 1].
        """
        raise NotImplementedError
