"""Regression coverage for saved variable-group state and Streamlit warnings."""

from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class VariableGroupStateTests(unittest.TestCase):
    def test_initialization_and_reruns_do_not_duplicate_widget_defaults(self):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
        # Streamlit logs this warning once per process; reset it for each run
        # so a previous widget cannot mask a later duplicate-default warning.
        for crop, group in (
            ("Potato", "Light response curve"),
            ("Potato", "Traits"),
            ("Potato", "Traits"),
            ("Tomato", "Weather"),
            ("Potato", "Traits"),
        ):
            with self.subTest(crop=crop, group=group):
                app.session_state["greenhouse__crop"] = crop
                if group != "Light response curve":
                    app.session_state[f"greenhouse__{crop.lower()}__variable_group"] = group
                with patch.object(policies, "_shown_default_value_warning", False):
                    with patch.object(policies._LOGGER, "warning") as warning:
                        app.run()
                self.assertFalse(app.exception)
                warning.assert_not_called()
                self.assertEqual(
                    app.session_state[f"greenhouse__{crop.lower()}__variable_group"], group
                )


if __name__ == "__main__":
    unittest.main()
