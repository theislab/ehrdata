"""Canonical table schemas for the non-OMOP source ingestion layer.

Each :class:`TableSchema` instance describes one output table produced by a
source adapter (MarketScan, LCED, CPRD).  They are the single source of truth
for column names, expected dtypes, and nullability across all adapters and
tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column in a canonical table.

    Args:
        name: Column name.
        dtype: Pandas dtype string (e.g. ``"object"``, ``"datetime64[ns]"``).
        nullable: Whether the column may contain ``NaN`` / ``NaT`` values.
    """

    name: str
    dtype: str
    nullable: bool = True


@dataclass(frozen=True)
class TableSchema:
    """Canonical schema for one output table.

    Args:
        name: Table name (used as key in :data:`ALL_SCHEMAS`).
        columns: Ordered column specifications.
    """

    name: str
    columns: tuple[ColumnSpec, ...]

    @property
    def column_names(self) -> list[str]:
        """Ordered list of column names."""
        return [c.name for c in self.columns]

    def empty(self) -> pd.DataFrame:
        """Return an empty :class:`~pandas.DataFrame` typed to this schema.

        Returns:
            Zero-row DataFrame with columns and dtypes matching the schema.
        """
        return pd.DataFrame({c.name: pd.Series(dtype=c.dtype) for c in self.columns})

    def validate(self, df: pd.DataFrame, *, strict: bool = False) -> list[str]:
        """Return validation error messages for *df* against this schema.

        Args:
            df: DataFrame to validate.
            strict: If ``True``, also report columns present in *df* but not in
                the schema.

        Returns:
            List of error strings; empty list means the DataFrame is valid.
        """
        errors: list[str] = []
        schema_cols = {c.name for c in self.columns}
        for col in self.columns:
            if col.name not in df.columns:
                errors.append(f"Missing required column '{col.name}'")
        if strict:
            extra = sorted(set(df.columns) - schema_cols)
            if extra:
                errors.append(f"Unexpected columns: {extra}")
        return errors


# ---------------------------------------------------------------------------
# Canonical table definitions
# ---------------------------------------------------------------------------

DIAGNOSIS = TableSchema(
    name="diagnosis",
    columns=(
        ColumnSpec("patient_id", "object", nullable=False),
        ColumnSpec("dxver", "object"),  # "0" = ICD-10, "9" = ICD-9, None = unknown
        ColumnSpec("eventdate", "datetime64[ns]"),
        ColumnSpec("dx", "object", nullable=False),
    ),
)

THERAPY = TableSchema(
    name="therapy",
    columns=(
        ColumnSpec("patient_id", "object", nullable=False),
        ColumnSpec("prescription_date", "datetime64[ns]"),
        ColumnSpec("start_date", "datetime64[ns]"),
        ColumnSpec("fill_date", "datetime64[ns]"),
        ColumnSpec("end_date", "datetime64[ns]"),
        ColumnSpec("refill", "Int64"),
        ColumnSpec("rxcui", "object"),
        ColumnSpec("ndc11", "object"),
        ColumnSpec("ingredient", "object"),
    ),
)

LABTEST = TableSchema(
    name="labtest",
    columns=(
        ColumnSpec("patient_id", "object", nullable=False),
        ColumnSpec("eventdate", "datetime64[ns]"),
        ColumnSpec("value", "object"),
        ColumnSpec("valuecat", "object"),
        ColumnSpec("unit", "object"),
        ColumnSpec("loinc", "object"),
    ),
)

PROCEDURE = TableSchema(
    name="procedure",
    columns=(
        ColumnSpec("patient_id", "object", nullable=False),
        ColumnSpec("proctype", "object"),
        ColumnSpec("eventdate", "datetime64[ns]"),
        ColumnSpec("proc", "object", nullable=False),
    ),
)

PATINFO = TableSchema(
    name="patinfo",
    columns=(
        ColumnSpec("patient_id", "object", nullable=False),
        ColumnSpec("dobyr", "Int64"),
        ColumnSpec("sex", "object"),
    ),
)

INSURANCE = TableSchema(
    name="insurance",
    columns=(
        ColumnSpec("patient_id", "object", nullable=False),
        ColumnSpec("svcdate", "datetime64[ns]"),
        ColumnSpec("cob", "float64"),  # coordination of benefits amount
        ColumnSpec("coins", "float64"),  # coinsurance amount
        ColumnSpec("copay", "float64"),
    ),
)

PROVIDER = TableSchema(
    name="provider",
    columns=(
        ColumnSpec("patient_id", "object", nullable=False),
        ColumnSpec("dtstart", "datetime64[ns]"),
        ColumnSpec("dtend", "datetime64[ns]"),
        ColumnSpec("plantyp", "object"),
        ColumnSpec("rx", "object"),  # pharmacy coverage flag
        ColumnSpec("hlthplan", "object"),
    ),
)

HABIT = TableSchema(
    name="habit",
    columns=(
        ColumnSpec("patient_id", "object", nullable=False),
        ColumnSpec("encounter_date", "datetime64[ns]"),
        ColumnSpec("mapped_question_answer", "object"),
    ),
)

ALL_SCHEMAS: dict[str, TableSchema] = {
    s.name: s for s in (DIAGNOSIS, THERAPY, LABTEST, PROCEDURE, PATINFO, INSURANCE, PROVIDER, HABIT)
}
