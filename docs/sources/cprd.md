# CPRD GOLD

CPRD GOLD (Clinical Practice Research Datalink) is a UK primary-care database derived
from anonymised GP records.  The extract described here is the May 2018 dementia cohort
(study DS059).  The Python adapter lives at `ehrdata.io.source.adapters.cprd`.

## Raw data

**Format:** Zip archives of tab-delimited `.txt` files.  Each file type (Clinical,
Referral, Test, Additional, Therapy, Consultation, Practice) is distributed as one
or more `.zip` files that are read directly without unpacking, using
`extract.read_zipped_tsvs`.

**Patient identifier:** `patid` — renamed to `patient_id` by the adapter.

**Date format:** `DD/MM/YYYY` — the adapter passes `formats=["%d/%m/%Y"]` to
`normalize.coerce_date`.

**Scale (May 2018 extract):**

| Category | Count |
|---|---|
| Total patients | 6,638,574 |
| Qualified patients | 6,613,198 |
| Total recordings | 3,726,733,860 |
| Total visits | 939,967,599 |

**Coding systems:**

| Domain | Column | Standard |
|---|---|---|
| Diagnosis | `medcode` | CPRD internal integer → Read code via `medical.txt` |
| Read code | `readcode` | Read code V2 (UK primary-care hierarchy) |
| Drug | `prodcode` | CPRD product dictionary → drug substance via `product.txt` |

CPRD does **not** use ICD codes.  The canonical `dxver` column is always `None` for
CPRD outputs.

## Source files

| File type | Key columns | Used for |
|---|---|---|
| Clinical | patid, eventdate, medcode, enttype, adid | diagnosis, labtest |
| Referral | patid, eventdate, medcode | diagnosis |
| Test | patid, eventdate, medcode, enttype, data1–data7 | diagnosis, labtest |
| Additional | patid, enttype, adid, data1–data7 | labtest (joined with Clinical) |
| Therapy | patid, eventdate, prodcode, bnfcode, qty, issueseq | therapy |
| Patient | patid, dobyr, sex, pracid | patinfo |
| Practice | pracid, region, lcd, uts | (reference; not in canonical output) |

## Vocabulary files

| File | Columns | Loaded by |
|---|---|---|
| `medical.txt` | medcode, readcode, desc | `vocab.readcode.load_medical_map` |
| `product.txt` | prodcode, drugsubstance, strength, … | `vocab.prodcode.load_product_map` |
| `product.csv` | prodcode, drugsubstance.updated, … | `vocab.prodcode.load_product_map` (preferred; uses `drugsubstance.updated`) |

`product.csv` is the derived file produced by CPRD's Preparation step, which normalises
multi-ingredient names.  When it is available, `load_product_map` uses
`drugsubstance.updated` automatically.

## Canonical output tables

| Canonical table | Sources | Notes |
|---|---|---|
| `diagnosis` | Clinical, Referral, Test | medcode translated to Read code when `medical_map` provided; `dxver = None` |
| `therapy` | Therapy | prodcode → drug substance when `product_map` provided; `prescription_date`, `start_date`, `end_date` = NaT; no NDC or RxCUI |
| `labtest` | Clinical+Additional (inner join on patid/adid/enttype) + Test | `data2`→value, `data3`→unit, `data4`→valuecat; `loinc = None` |
| `patinfo` | Patient | Three canonical columns only (patient_id, dobyr, sex) |

## Lab test join

CPRD stores lab values across two complementary file types:

- **Additional** files hold the numeric values (`data1`–`data7`) keyed by `(patid, adid, enttype)`
  but contain no `eventdate`.
- **Clinical** files hold `eventdate` keyed by the same composite key.

`build_labtest` performs an inner join of Clinical and Additional on
`(patid, adid, enttype)` to recover `eventdate` for each data row, then unions with
Test files (which already contain both `eventdate` and data columns).

## Usage

```python
from ehrdata.io.source.adapters import cprd
from ehrdata.io.source.vocab import readcode, prodcode

medical_map = readcode.load_medical_map("medical.txt")
product_map = prodcode.load_product_map("product.csv")   # or product.txt

diag    = cprd.build_diagnosis(clinical, referral, test_data, medical_map=medical_map)
therapy = cprd.build_therapy(therapy_data, product_map=product_map)
labs    = cprd.build_labtest(clinical, additional, test_data)
patinfo = cprd.build_patinfo(patient)
```

## Notes

- Read codes starting with `R` (Symptoms, Signs and Ill-Defined Conditions) are not
  disease codes and are typically excluded from phenotype analyses (filter
  `~dx.str.startswith("R")`).
- Read codes starting with `ZZ` could not be mapped to any hierarchy in the original
  extract and should be investigated before use.
- `issueseq` in the Therapy file is a repeat-prescription counter (1 = first issue,
  2 = first repeat, etc.).  It is not mapped to the canonical `refill` column.
- The `practice` table provides region, last-collection date (`lcd`), and
  up-to-standard date (`uts`) for each GP practice.  These are useful for defining
  study observation windows but are not included in the canonical `patinfo` output.
