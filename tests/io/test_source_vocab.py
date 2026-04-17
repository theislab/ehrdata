from pathlib import Path

import pandas as pd
import pytest

from ehrdata.io.source.vocab.ndc import join_ingredient_by_ndc, load_ndc_ingredient_map
from ehrdata.io.source.vocab.rxnorm import join_ingredient_by_rxcui, load_rxcui_ingredient_map

VOCAB_DIR = Path("tests/data/source_vocab")


# ---------------------------------------------------------------------------
# NDC
# ---------------------------------------------------------------------------


class TestLoadNdcIngredientMap:
    @pytest.fixture()
    def ndc_map(self):
        return load_ndc_ingredient_map(VOCAB_DIR / "ndc_ingredient_map.txt")

    def test_returns_dataframe(self, ndc_map):
        assert isinstance(ndc_map, pd.DataFrame)

    def test_columns(self, ndc_map):
        assert list(ndc_map.columns) == ["ndc11", "rxcui", "ingredient"]

    def test_row_count(self, ndc_map):
        assert len(ndc_map) == 10

    def test_ndc11_zero_padded(self, ndc_map):
        # All ndc11 values must be exactly 11 characters
        assert ndc_map["ndc11"].str.len().eq(11).all()

    def test_ndc11_is_string(self, ndc_map):
        assert ndc_map["ndc11"].dtype == object

    def test_known_ingredient(self, ndc_map):
        row = ndc_map[ndc_map["ndc11"] == "00071015523"]
        assert row["ingredient"].iloc[0] == "metformin"

    def test_leading_zeros_preserved(self, ndc_map):
        # 00065063136 starts with two zeros — must not be dropped
        assert "00065063136" in ndc_map["ndc11"].values

    def test_accepts_string_path(self):
        df = load_ndc_ingredient_map(str(VOCAB_DIR / "ndc_ingredient_map.txt"))
        assert len(df) == 10


class TestJoinIngredientByNdc:
    @pytest.fixture()
    def ndc_map(self):
        return load_ndc_ingredient_map(VOCAB_DIR / "ndc_ingredient_map.txt")

    def test_matched_row_gets_ingredient(self, ndc_map):
        df = pd.DataFrame({"patient_id": ["1"], "ndc11": ["00071015523"]})
        result = join_ingredient_by_ndc(df, ndc_map)
        assert result.loc[0, "ingredient"] == "metformin"

    def test_unmatched_row_gets_nan(self, ndc_map):
        df = pd.DataFrame({"patient_id": ["1"], "ndc11": ["99999999999"]})
        result = join_ingredient_by_ndc(df, ndc_map)
        assert pd.isna(result.loc[0, "ingredient"])

    def test_existing_ingredient_column_overwritten(self, ndc_map):
        df = pd.DataFrame({"patient_id": ["1"], "ndc11": ["00071015523"], "ingredient": ["old"]})
        result = join_ingredient_by_ndc(df, ndc_map)
        assert result.loc[0, "ingredient"] == "metformin"

    def test_row_count_preserved(self, ndc_map):
        df = pd.DataFrame({"patient_id": ["1", "2", "3"], "ndc11": ["00071015523", "99999999999", "00310751030"]})
        result = join_ingredient_by_ndc(df, ndc_map)
        assert len(result) == 3

    def test_custom_ndc_column_name(self, ndc_map):
        df = pd.DataFrame({"patient_id": ["1"], "drug_ndc": ["00071015523"]})
        result = join_ingredient_by_ndc(df, ndc_map, ndc_col="drug_ndc")
        assert result.loc[0, "ingredient"] == "metformin"

    def test_does_not_mutate_input(self, ndc_map):
        df = pd.DataFrame({"ndc11": ["00071015523"]})
        _ = join_ingredient_by_ndc(df, ndc_map)
        assert "ingredient" not in df.columns

    def test_duplicate_ndc_in_map_no_explosion(self, ndc_map):
        # Even if map has duplicates, join must not multiply rows
        dup_map = pd.concat([ndc_map, ndc_map.head(1)], ignore_index=True)
        df = pd.DataFrame({"ndc11": ["00071015523", "00310751030"]})
        result = join_ingredient_by_ndc(df, dup_map)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# RxNorm
# ---------------------------------------------------------------------------


class TestLoadRxcuiIngredientMap:
    @pytest.fixture()
    def rxcui_map(self):
        return load_rxcui_ingredient_map(VOCAB_DIR / "rxcui_ingredient_map.txt")

    def test_returns_dataframe(self, rxcui_map):
        assert isinstance(rxcui_map, pd.DataFrame)

    def test_columns(self, rxcui_map):
        assert list(rxcui_map.columns) == ["rxcui", "ingredient"]

    def test_v1_index_column_dropped(self, rxcui_map):
        assert "v1" not in rxcui_map.columns

    def test_row_count(self, rxcui_map):
        assert len(rxcui_map) == 10

    def test_known_ingredient(self, rxcui_map):
        row = rxcui_map[rxcui_map["rxcui"] == "723"]
        assert row["ingredient"].iloc[0] == "metformin"

    def test_rxcui_is_string(self, rxcui_map):
        assert rxcui_map["rxcui"].dtype == object


class TestJoinIngredientByRxcui:
    @pytest.fixture()
    def rxcui_map(self):
        return load_rxcui_ingredient_map(VOCAB_DIR / "rxcui_ingredient_map.txt")

    def test_matched_row_gets_ingredient(self, rxcui_map):
        df = pd.DataFrame({"patient_id": ["1"], "rxcui": ["723"]})
        result = join_ingredient_by_rxcui(df, rxcui_map)
        assert result.loc[0, "ingredient"] == "metformin"

    def test_unmatched_row_gets_nan(self, rxcui_map):
        df = pd.DataFrame({"patient_id": ["1"], "rxcui": ["9999999"]})
        result = join_ingredient_by_rxcui(df, rxcui_map)
        assert pd.isna(result.loc[0, "ingredient"])

    def test_existing_ingredient_column_overwritten(self, rxcui_map):
        df = pd.DataFrame({"rxcui": ["723"], "ingredient": ["old"]})
        result = join_ingredient_by_rxcui(df, rxcui_map)
        assert result.loc[0, "ingredient"] == "metformin"

    def test_row_count_preserved(self, rxcui_map):
        df = pd.DataFrame({"rxcui": ["723", "9999999", "4815"]})
        result = join_ingredient_by_rxcui(df, rxcui_map)
        assert len(result) == 3

    def test_custom_rxcui_column_name(self, rxcui_map):
        df = pd.DataFrame({"drug_rxcui": ["723"]})
        result = join_ingredient_by_rxcui(df, rxcui_map, rxcui_col="drug_rxcui")
        assert result.loc[0, "ingredient"] == "metformin"

    def test_does_not_mutate_input(self, rxcui_map):
        df = pd.DataFrame({"rxcui": ["723"]})
        _ = join_ingredient_by_rxcui(df, rxcui_map)
        assert "ingredient" not in df.columns
