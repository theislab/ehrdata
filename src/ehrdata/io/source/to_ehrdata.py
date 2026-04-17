"""Bridge from canonical source tables to :class:`~ehrdata.EHRData`.

Converts the flat event DataFrames produced by the source adapters
(MarketScan, LCED, CPRD) into a binary presence matrix in an
:class:`~ehrdata.EHRData` object.

Mapping
-------
- **obs** — one row per patient, taken from the canonical PATINFO table
  (``patient_id`` becomes the index; ``dobyr`` and ``sex`` are kept as
  annotation columns).
- **var** — one row per unique clinical concept encountered across all
  supplied event tables.  The index is formatted as ``"{source}:{code}"``
  (e.g. ``"diagnosis:E11.9"``, ``"therapy:metformin"``, ``"labtest:14749-6"``)
  so that codes from different coding systems never collide.  Columns
  ``concept_source`` and ``concept_code`` unpack the index for easy querying.
- **X** — ``float64`` binary presence matrix of shape ``(n_obs × n_var)``:
  ``1.0`` when a patient has at least one event for that concept, ``0.0``
  otherwise.  Patients absent from *patinfo* are silently excluded.
  Duplicate events do not inflate the value beyond ``1.0``.
- **uns** — provenance metadata keyed ``source_io_source`` (the *source*
  argument) and ``source_io_tables`` (list of table names that were provided).

This function does **not** build a time dimension (R tensor).  Users who need
interval-level temporal encoding should aggregate the canonical DataFrames
into time bins before calling this function, or use the OMOP
``setup_variables`` path for OMOP-formatted data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ehrdata import EHRData


def to_ehrdata(
    patinfo: pd.DataFrame,
    *,
    diagnosis: pd.DataFrame | None = None,
    therapy: pd.DataFrame | None = None,
    labtest: pd.DataFrame | None = None,
    procedure: pd.DataFrame | None = None,
    source: str | None = None,
) -> EHRData:
    """Convert canonical source tables into an :class:`~ehrdata.EHRData` presence matrix.

    Args:
        patinfo: Canonical PATINFO DataFrame (must include ``patient_id``,
            ``dobyr``, ``sex``).  Defines the observation set — only patients
            present here appear as rows in the output.
        diagnosis: Optional canonical DIAGNOSIS DataFrame.  The ``dx`` column
            supplies concept codes prefixed with ``"diagnosis:"``.
        therapy: Optional canonical THERAPY DataFrame.  The ``ingredient``
            column supplies concept codes prefixed with ``"therapy:"``.  Rows
            with null ``ingredient`` are skipped.
        labtest: Optional canonical LABTEST DataFrame.  The ``loinc`` column
            supplies concept codes prefixed with ``"labtest:"``.  Rows with
            null ``loinc`` are skipped.
        procedure: Optional canonical PROCEDURE DataFrame.  The ``proc``
            column supplies concept codes prefixed with ``"procedure:"``.
        source: Optional free-text label for the data source (e.g.
            ``"marketscan"``, ``"lced"``, ``"cprd"``).  Stored in
            ``uns["source_io_source"]``.

    Returns:
        :class:`~ehrdata.EHRData` with ``obs``, ``var``, ``X``, and ``uns``
        populated.  When no event tables are provided, ``X`` has zero columns
        and ``var`` is empty.

    Examples:
        >>> import pandas as pd
        >>> import ehrdata as ed
        >>> patinfo = pd.DataFrame({"patient_id": ["P1", "P2"], "dobyr": [1960, 1975], "sex": ["M", "F"]})
        >>> diagnosis = pd.DataFrame(
        ...     {
        ...         "patient_id": ["P1", "P1", "P2"],
        ...         "dx": ["E11.9", "I10", "E11.9"],
        ...         "dxver": [None, None, None],
        ...         "eventdate": pd.NaT,
        ...     }
        ... )
        >>> edata = ed.io.source.to_ehrdata(patinfo, diagnosis=diagnosis, source="example")
        >>> edata.obs_names.tolist()
        ['P1', 'P2']
        >>> "diagnosis:E11.9" in edata.var_names
        True
    """
    from ehrdata import EHRData

    # ---- obs: one row per patient ----------------------------------------
    obs = patinfo[["patient_id", "dobyr", "sex"]].copy()
    obs = obs.drop_duplicates(subset=["patient_id"])
    obs = obs.set_index("patient_id")
    obs.index = obs.index.astype(str)
    obs.index.name = "patient_id"
    patients: list[str] = obs.index.tolist()
    patient_set: set[str] = set(patients)
    patient_idx: dict[str, int] = {p: i for i, p in enumerate(patients)}

    # ---- collect (patient_id, concept) presence pairs --------------------
    concept_frames: list[pd.DataFrame] = []
    tables_used: list[str] = []

    if diagnosis is not None:
        _append_pairs(concept_frames, diagnosis, "patient_id", "dx", "diagnosis")
        tables_used.append("diagnosis")

    if therapy is not None:
        _append_pairs(concept_frames, therapy, "patient_id", "ingredient", "therapy")
        tables_used.append("therapy")

    if labtest is not None:
        _append_pairs(concept_frames, labtest, "patient_id", "loinc", "labtest")
        tables_used.append("labtest")

    if procedure is not None:
        _append_pairs(concept_frames, procedure, "patient_id", "proc", "procedure")
        tables_used.append("procedure")

    uns: dict = {
        "source_io_source": source,
        "source_io_tables": tables_used,
    }

    if not concept_frames:
        var = pd.DataFrame(
            {"concept_source": pd.Series(dtype=object), "concept_code": pd.Series(dtype=object)},
        )
        var.index.name = "concept"
        X = np.zeros((len(patients), 0), dtype=np.float64)
        return EHRData(X=X, obs=obs, var=var, uns=uns)

    # ---- deduplicate and restrict to known patients -----------------------
    all_pairs = pd.concat(concept_frames, ignore_index=True).drop_duplicates()
    all_pairs["patient_id"] = all_pairs["patient_id"].astype(str)
    all_pairs = all_pairs[all_pairs["patient_id"].isin(patient_set)]

    # ---- var: one row per unique concept ----------------------------------
    concepts: list[str] = sorted(all_pairs["concept"].unique())
    concept_idx: dict[str, int] = {c: i for i, c in enumerate(concepts)}

    var = pd.DataFrame(
        {
            "concept_source": pd.array([c.split(":", 1)[0] for c in concepts], dtype=object),
            "concept_code": pd.array([c.split(":", 1)[1] for c in concepts], dtype=object),
        },
        index=pd.Index(concepts, name="concept"),
    )

    # ---- X: binary presence matrix (obs × var) ---------------------------
    pid_indices = all_pairs["patient_id"].map(patient_idx)
    con_indices = all_pairs["concept"].map(concept_idx)

    valid = pid_indices.notna() & con_indices.notna()
    rows = pid_indices[valid].astype(int).values
    cols = con_indices[valid].astype(int).values

    X = np.zeros((len(patients), len(concepts)), dtype=np.float64)
    X[rows, cols] = 1.0

    return EHRData(X=X, obs=obs, var=var, uns=uns)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _append_pairs(
    frames: list[pd.DataFrame],
    df: pd.DataFrame,
    pid_col: str,
    code_col: str,
    prefix: str,
) -> None:
    """Extract (patient_id, prefixed concept) pairs from *df* and append to *frames*."""
    sub = df[[pid_col, code_col]].dropna(subset=[code_col])
    if sub.empty:
        return
    frames.append(
        pd.DataFrame(
            {
                "patient_id": sub[pid_col].astype(str).values,
                "concept": prefix + ":" + sub[code_col].astype(str).values,
            }
        )
    )
