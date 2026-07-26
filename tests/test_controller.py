import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import controller


class ControllerTests(unittest.TestCase):
    def test_no_command_prints_the_beginner_guide(self):
        with patch("sys.argv", ["controller.py"]), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(controller.main(), 0)

        self.assertIn("python controller.py all", output.getvalue())
        self.assertIn("predict_configurable", output.getvalue())

    def test_predict_uses_the_default_input_folder(self):
        with patch("controller.run", return_value=0) as run:
            controller.predict(None)

        run.assert_called_once_with(
            "-m", "predicting.predict_configurable", "predicting/images"
        )

    def test_train_passes_the_overlay_flag(self):
        with patch("controller.run", return_value=0) as run:
            controller.train(with_overlays=True)

        run.assert_called_once_with("-m", "training.train", "--with-overlays")

    def test_all_stops_when_a_step_fails(self):
        with (
            patch("controller.validate", return_value=0) as validate,
            patch("controller.train", return_value=1) as train,
            patch("controller.predict") as predict,
            patch("controller.statistics") as statistics,
            patch("controller.analysis") as analysis,
        ):
            self.assertEqual(controller.all_steps(), 1)

        validate.assert_called_once_with(False)
        train.assert_called_once_with(False)
        predict.assert_not_called()
        statistics.assert_not_called()
        analysis.assert_not_called()


if __name__ == "__main__":
    unittest.main()
