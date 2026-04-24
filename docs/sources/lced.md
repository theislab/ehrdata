# IBM LCED (Limited Claims-EMR Data)

IBM LCED (Limited Claims-EMR Data) is a linked dataset that combines IBM MarketScan
claims with Explorys EMR data.  It is a superset of MarketScan: every patient in
LCED also has claims records, and a subset additionally has structured EMR data from
the Explorys network.  The Python adapter lives at
`ehrdata.io.source.adapters.lced`.

## Raw data

**Format:** PostgreSQL database, views prefixed with `v_`.

**Patient identifier:** `patient_id` — already correct; no rename needed.

**Scale (original extract):**

| Table | Patients | Observations |
|---|---|---|
| v_drug | 2,614,657 | 132,831,286 |
| v_encounter | 3,630,723 | 191,489,491 |
| v_habit | 2,004,540 | 42,745,066 |
| v_observation | 2,952,436 | 1,249,354,089 |
| v_annual_summary_enrollment | 3,675,923 | 11,755,699 |
| v_detail_enrollment | 3,675,923 | 115,140,511 |
| v_facility_header | 2,496,039 | 127,987,056 |
| v_inpatient_admissions | 606,967 | 1,078,418 |
| v_inpatient_services | 606,967 | 38,790,918 |
| v_lab_results | 394,602 | 42,530,560 |
| v_outpatient_drug_claims | 2,682,183 | 143,163,508 |
| v_outpatient_services | 3,281,396 | 365,200,712 |
| **Total** | **4,367,831** | **3,057,791,040** |

**Coding systems:**

| Domain | Column | Standard | Source |
|---|---|---|---|
| Diagnosis | `dx1`–`dx9`+ | ICD-9-CM / ICD-10-CM | MarketScan views |
| Diagnosis version | `dxver` | `"9"` / `"0"`; frequently null | MarketScan views |
| Drug (EMR) | `rx_cui` | RxNorm CUI | v_drug (Explorys) |
| Drug (claims) | `ndcnum` | NDC-10 | v_outpatient_drug_claims |
| Lab (EMR) | `loinc_test_id` | LOINC | v_observation (Explorys) |
| Lab (claims) | `loinccd` | LOINC | v_lab_results |

**Key data-source notes (from original ETL):**

- Diagnosis information is only available in MarketScan views, not Explorys.
- `v_drug.rx_cui` has far fewer missing values than NDC codes; use it when available.
- `dxver` has extensive missingness; ICD-9 is inferred from E/V prefix by the adapter.
- `pdx` in inpatient tables is the same as `dx1`; it is not double-counted.
- `v_lab_results.resltcat` (categorical result) is only 3.1% non-missing and is preserved as `valuecat`.
- Habit data (`v_habit`) is Explorys-only and is not available in plain MarketScan.
- `v_encounter.encounter_date` is always null; only `encounter_join_id` is used to join habits.

## Source tables and adapter mapping

### Diagnosis

| Source view | Key columns |
|---|---|
| v_facility_header | patient_id, svcdate, dx1–dx9, dxver |
| v_inpatient_admissions | patient_id, admdate, pdx, dx1–dx15, dxver |
| v_inpatient_services | patient_id, svcdate, pdx, dx1–dx4, dxver |
| v_lab_results | patient_id, svcdate, dx1, dxver |
| v_outpatient_services | patient_id, svcdate, dx1–dx4, dxver |

Wide dx columns are unnested to one row per code; `dxver` backfilled for E/V-prefix codes.

### Therapy

Two sources are joined separately before union because they use different drug coding systems:

| Source view | Key columns | Drug identifier |
|---|---|---|
| v_drug (Explorys) | patient_id, prescription_date, start_date, end_date, rx_cui | RxNorm → ingredient via `vocab/rxnorm.py` |
| v_outpatient_drug_claims (MarketScan) | patient_id, svcdate, daysupp, refill, ndcnum | NDC-11 → ingredient via `vocab/ndc.py` |

### Lab test

| Source view | Key columns | Mapping |
|---|---|---|
| v_observation (Explorys) | patient_id, observation_date, std_value, std_uom, loinc_test_id | → eventdate, value, unit, loinc |
| v_lab_results (MarketScan) | patient_id, svcdate, result, resltcat, resunit, loinccd | → eventdate, value, valuecat, unit, loinc |

### Habit

`v_habit` (patient_id, mapped_question_answer, encounter_join_id) is LEFT JOINed with
`v_encounter` (patient_id, encounter_date, encounter_join_id) to recover the event date.
Rows with null `mapped_question_answer` are dropped.

### Other tables

`patinfo`, `insurance`, and `provider` follow the same pattern as MarketScan.

## Usage

```python
from ehrdata.io.source.adapters import lced
from ehrdata.io.source.vocab import ndc, rxnorm, loinc

ndc_map   = ndc.load_ndc_ingredient_map("ndc_ingredient_map.txt")
rxcui_map = rxnorm.load_rxcui_ingredient_map("rxcui_ingredient_map.txt")
loinc_map = loinc.load_loinc_map("loinc_map.csv")

diag    = lced.build_diagnosis(
    facility_header, inpatient_admissions, inpatient_services,
    lab_results, outpatient_services,
)
therapy = lced.build_therapy(
    v_drug, v_outpatient_drug_claims, ndc_map=ndc_map, rxcui_map=rxcui_map
)
labs    = lced.build_labtest(v_observation, v_lab_results, loinc_map=loinc_map)
habit   = lced.build_habit(v_habit, v_encounter)
patinfo = lced.build_patinfo(v_annual_summary_enrollment)
```
