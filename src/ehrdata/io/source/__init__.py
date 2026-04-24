"""Non-OMOP source ingestion layer.

Provides canonical schema definitions, generic extraction helpers, and
normalization pipelines for claims and EHR data sources (MarketScan, LCED,
CPRD).  Adapters for individual data sources live under
:mod:`ehrdata.io.source.adapters`.

Typical usage::

    from ehrdata.io import source

    df = source.extract.union_tables([df1, df2])
    df = source.normalize.normalize_diagnosis(df)
    errors = source.schema.DIAGNOSIS.validate(df)
"""

from . import adapters, extract, normalize, schema, vocab
from .schema import ALL_SCHEMAS, TableSchema
from .to_ehrdata import to_ehrdata

__all__ = [
    "ALL_SCHEMAS",
    "TableSchema",
    "adapters",
    "extract",
    "normalize",
    "schema",
    "to_ehrdata",
    "vocab",
]
