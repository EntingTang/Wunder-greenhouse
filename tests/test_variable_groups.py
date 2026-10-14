"""Playwright coverage against a running app (GREENHOUSE_BASE_URL overrides 8501)."""

from io import BytesIO
import os
from pathlib import Path
import unittest

import pandas as pd
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OJIP_LABELS = [
    "Fo_OJIP", "F300", "FJ", "FI", "Fm_OJIP", "Fv/Fm_OJIP", "Mo", "VK", "VJ",
    "VI", "VK_VJ", "phi_Po", "psi_o", "phi_Eo", "RC_ABS", "PIabs", "PItot",
]


class VariableGroupUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.context = self.browser.new_context(accept_downloads=True)
        self.page = self.context.new_page()
        self.page.set_default_timeout(30_000)
        self.page.goto(os.environ.get("GREENHOUSE_BASE_URL", "http://localhost:8501"))
        expect(self.page.get_by_role("heading", name="Response variable", exact=True)).to_be_visible()

    def tearDown(self):
        self.context.close()

    def select_crop(self, crop):
        self.page.get_by_role("radio", name=crop, exact=True).check()
        expect(self.page.get_by_role("heading", name=crop, exact=True)).to_be_visible()

    def select_group(self, group):
        self.page.get_by_role("button", name=group, exact=True).click()

    def response_labels(self):
        labels = self.page.get_by_role("radiogroup", name="Response variable", exact=True).locator(
            "label"
        ).all_text_contents()
        return [label.strip() for label in labels]

    def select_variable(self, label):
        self.page.get_by_role("radiogroup", name="Response variable", exact=True).get_by_role(
            "radio", name=label, exact=True
        ).check()

    def expect_axes(self, y_label, *, time_series=True, series_name=None):
        self.page.wait_for_function(
            """({yLabel, timeSeries, seriesName}) => {
                const plot = document.querySelector('.js-plotly-plot');
                return plot?._fullLayout?.yaxis?.title?.text === yLabel
                    && plot._fullLayout.xaxis.title.text ===
                        (timeSeries ? 'Measurement date' : 'PAR (µmol m⁻² s⁻¹)')
                    && plot._fullLayout.xaxis.type === (timeSeries ? 'date' : 'linear')
                    && (!seriesName || plot._fullData.some(trace => trace.name === seriesName));
            }""",
            arg={"yLabel": y_label, "timeSeries": time_series, "seriesName": series_name},
        )
        expect(self.page.get_by_test_id("stException")).to_have_count(0)
        expect(self.page.get_by_text("Static export could not be prepared:", exact=False)).to_have_count(0)

    def test_light_response_order_and_par_axis_for_both_crops(self):
        for crop in ("Potato", "Tomato"):
            with self.subTest(crop=crop):
                self.select_crop(crop)
                self.select_group("Light response curve")
                self.expect_axes("Y(II)", time_series=False)
                self.assertEqual(self.response_labels()[:3], ["F", "Fm′", "Fo′"])
                self.select_variable("Fo′")
                self.expect_axes("Fo′", time_series=False)

    def test_group_members_labels_units_and_restored_selection(self):
        for crop in ("Potato", "Tomato"):
            with self.subTest(crop=crop):
                self.select_crop(crop)
                groups = [
                    ("OJIP curve", OJIP_LABELS, "Fv/Fm_OJIP", "Fv/Fm_OJIP"),
                    (
                        "Traits",
                        ["Fv/Fm_LSA", "ChlAvg", "AFLAV", "NBI"]
                        + (["gs_Avg"] if crop == "Potato" else []) + ["PlantHeight"],
                        "ChlAvg", "ChlAvg (mg m⁻²)",
                    ),
                    (
                        "Weather", ["PAR", "SolarRadiation", "WindSpeed", "Ta", "VPD"],
                        "SolarRadiation", "SolarRadiation (W m⁻²)",
                    ),
                    ("Soil", ["SoilMoisture"], "SoilMoisture", "SoilMoisture (m³ m⁻³)"),
                ]
                for group, labels, selection, y_label in groups:
                    self.select_group(group)
                    expect(self.page.get_by_role("radio", name=labels[0], exact=True)).to_be_visible()
                    self.assertEqual(self.response_labels(), labels)
                    self.select_variable(selection)
                    self.expect_axes(y_label)
                self.select_group("Traits")
                self.expect_axes("ChlAvg (mg m⁻²)")
                expect(self.page.get_by_role("radio", name="ChlAvg", exact=True)).to_be_checked()

    def test_treatment_time_series_preserves_records_without_par(self):
        files = {
            "Potato": "Potato_meteo_HH2_LSA_biomass_OJIP_tillHarvest_LSA_updated.xlsx",
            "Tomato": "LMM_tomato_input_tillHarvest_LSA_updated.xlsx",
        }
        for crop, filename in files.items():
            with self.subTest(crop=crop):
                self.select_crop(crop)
                self.select_group("Weather")
                self.select_variable("SolarRadiation")
                self.page.get_by_role("radio", name="Compare Data", exact=True).check()
                self.page.get_by_role("radio", name="Water treatments", exact=True).check()
                self.expect_axes("SolarRadiation (W m⁻²)", series_name="WW")
                self.page.get_by_text("Plot data and downloads", exact=True).click()
                with self.page.expect_download() as download_info:
                    self.page.get_by_role("button", name="Download CSV", exact=True).click()
                actual = pd.read_csv(BytesIO(Path(download_info.value.path()).read_bytes()))
                raw = pd.read_excel(ROOT / "data" / filename)
                self.assertTrue(raw["PAR(umol_m2_s1)"].isna().any())
                expected = raw.groupby(["Date", "Water", "Nitrogen"], observed=True).agg(
                    Mean=("SolarRadiation(W_m2)", "mean"),
                    SD=("SolarRadiation(W_m2)", "std"),
                    n=("SolarRadiation(W_m2)", "count"),
                ).reset_index()
                actual = actual.rename(columns={"SolarRadiation mean": "Mean", "SolarRadiation SD": "SD"})
                actual["Date"] = pd.to_datetime(actual["Date"])
                columns = ["Date", "Water", "Nitrogen", "Mean", "SD", "n"]
                order = ["Date", "Water", "Nitrogen"]
                pd.testing.assert_frame_equal(
                    actual[columns].sort_values(order).reset_index(drop=True),
                    expected[columns].sort_values(order).reset_index(drop=True),
                    check_dtype=False, rtol=1e-8,
                )
                self.assertNotIn("PAR", actual.columns)
                self.page.get_by_role("radio", name="Nitrogen treatments", exact=True).check()
                self.expect_axes("SolarRadiation (W m⁻²)", series_name="N0")


if __name__ == "__main__":
    unittest.main()
