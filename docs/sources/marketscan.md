# IBM MarketScan

IBM MarketScan is a US administrative claims database covering commercially insured
and Medicare-eligible populations.  The Python adapter lives at
`ehrdata.io.source.adapters.marketscan`.

## Raw data

**Format:** PostgreSQL database (schema `commercial`).  Each table is a view over
the underlying claims warehouse.

**Patient identifier:** `enrolid` (bigint) — renamed to `patient_id` by the adapter.

**Date columns:** standard `DATE` type; no format conversion needed.

**Coding systems:**

| Domain | Column | Standard |
|---|---|---|
| Diagnosis | `dx1`–`dx15` | ICD-9-CM / ICD-10-CM |
| Diagnosis version | `dxver` | `"9"` = ICD-9, `"0"` = ICD-10; often null for ICD-9 E/V codes |
| Drug | `ndcnum` | NDC-10 (zero-padded to NDC-11 by the adapter) |
| Procedure | `proc1`–`proc15` | CPT, HCPCS, ICD procedure codes |

## Source tables

| Table | Key columns | Used for |
|---|---|---|
| `facility_header` | enrolid, dxver, svcdate, dx1–dx9, proc1–proc6, cob, coins, copay | diagnosis, procedure, insurance |
| `inpatient_admissions` | enrolid, dxver, admdate, pdx, dx1–dx15, pproc, proc1–proc15 | diagnosis, procedure |
| `inpatient_services` | enrolid, dxver, svcdate, pdx, dx1–dx4, proctyp, pdx, proc1, cob, coins, copay | diagnosis, procedure, insurance |
| `outpatient_services` | enrolid, dxver, svcdate, dx1–dx4, proctyp, proc1, cob, coins, copay | diagnosis, procedure, insurance |
| `outpatient_prescription_drugs` | enrolid, svcdate, daysupp, refill, ndcnum, cob, coins, copay | therapy, insurance |
| `enrollment_annual_summary` | enrolid, dobyr, sex, efamid, year, region, … | patinfo |
| `enrollment_detail` | enrolid, dobyr, sex, dtstart, dtend, plantyp, rx, hlthplan, … | patinfo, provider |

## Canonical output tables

| Canonical table | Sources | Notes |
|---|---|---|
| `diagnosis` | facility_header, inpatient_admissions, inpatient_services, outpatient_services | Wide dx columns unnested to one row per code; ICD-9 inferred for E/V-prefix codes with null `dxver` |
| `therapy` | outpatient_prescription_drugs | `ndcnum` zero-padded to `ndc11`; `end_date = fill_date + daysupp`; optional NDC→ingredient join |
| `procedure` | facility_header, inpatient_admissions, inpatient_services, outpatient_services | `inpatient_services` unnests `[pdx, proc1]` only, matching the original ETL |
| `patinfo` | enrollment_annual_summary + 6 other tables | Extra MarketScan columns (efamid, year, region, …) preserved when present across all inputs |
| `insurance` | facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services | cob / coins / copay coerced to float64 |
| `provider` | enrollment_detail | DISTINCT on dtstart, dtend, plantyp, rx, hlthplan |

## Usage

```python
import pandas as pd
from ehrdata.io.source.adapters import marketscan
from ehrdata.io.source.vocab import ndc

# Load vocab (optional)
ndc_map = ndc.load_ndc_ingredient_map("path/to/ndc_ingredient_map.txt")

# Build canonical tables
diag = marketscan.build_diagnosis(
    facility_header, inpatient_admissions, inpatient_services, outpatient_services
)
therapy = marketscan.build_therapy(outpatient_prescription_drugs, ndc_map=ndc_map)
proc = marketscan.build_procedure(
    facility_header, inpatient_admissions, inpatient_services, outpatient_services
)
patinfo = marketscan.build_patinfo(
    enrollment_annual_summary, enrollment_detail,
    facility_header, inpatient_admissions,
    inpatient_services, outpatient_prescription_drugs, outpatient_services,
)
```

## Notes

- MarketScan does not include EMR lifestyle data (`habit` table is absent; use LCED for that).
- `dxver` is frequently null for ICD-9 codes that begin with `E` or `V`.
  The adapter calls `normalize.infer_icd_version` to backfill these automatically.
- `ndcnum` in the raw table is NDC-10 (may be fewer than 11 digits).
  The adapter zero-pads it to NDC-11 before any ingredient join.
