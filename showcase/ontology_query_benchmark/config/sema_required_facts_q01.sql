-- Required-fact ledger for milestone A question Q01:
--   "How many living patients are female (administrative gender)?"
-- Independently written; uses no SEMA importer or projection helper.
-- Run on SQL warehouse feedface00000000 (partner TEST). Every statement returns
-- aggregates only; none returns a patient identifier.
-- Origin of each fact:
--   supplied   = asserted by the CPD v2 product spec (operator-supplied binding)
--   discovered = read here from the Standard Library or the catalog

-- F1 (supplied by CPD, checked here against the catalog): the demographics product
-- has the gender and vital-status value columns and their label companions.
-- Expect 4 rows, all data_type STRING.
SELECT column_name, data_type
FROM partner_dev.information_schema.columns
WHERE table_schema = 'partner_provenance'
  AND table_name = 'pb_demographics_person_r3_platinum'
  AND column_name IN (
    'administrative_gender_PRTL2140000129', 'administrative_gender_PRTL2140000129_label',
    'vital_status_PRT3009506', 'vital_status_PRT3009506_label')
ORDER BY column_name;

-- F2 (discovered from the Standard Library): Female is PRT8532 under data element
-- PRTL2140000129 (Administrative gender); Alive is PRT45884343 under PRT3009506
-- (Vital status). Expect exactly these two rows.
SELECT data_element_prt_id, permissible_value_prt_id, permissible_value_prt_preferred_label
FROM partner_library_test.standard_library.standard_library_data_element
WHERE (data_element_prt_id = 'PRTL2140000129'
       AND lower(permissible_value_prt_preferred_label) = 'female')
   OR (data_element_prt_id = 'PRT3009506'
       AND lower(permissible_value_prt_preferred_label) = 'alive')
ORDER BY data_element_prt_id;

-- F3 (supplied by CPD as pv_id_column = value column; verified here): each value column
-- holds Standard Library permissible-value ids, with the label in the companion.
-- Expect not_in_library = 0 for every non-null value and one label per value.
SELECT 'administrative_gender' AS element, p.administrative_gender_PRTL2140000129 AS value,
       min(p.administrative_gender_PRTL2140000129_label) AS label,
       count(DISTINCT p.administrative_gender_PRTL2140000129_label) AS labels_per_value,
       count(*) AS row_count,
       max(CASE WHEN l.permissible_value_prt_id IS NULL
                 AND p.administrative_gender_PRTL2140000129 IS NOT NULL
                THEN 1 ELSE 0 END) AS not_in_library
FROM partner_dev.partner_provenance.pb_demographics_person_r3_platinum p
LEFT JOIN partner_library_test.standard_library.standard_library_data_element l
  ON l.data_element_prt_id = 'PRTL2140000129'
 AND l.permissible_value_prt_id = p.administrative_gender_PRTL2140000129
GROUP BY p.administrative_gender_PRTL2140000129
UNION ALL
SELECT 'vital_status', p.vital_status_PRT3009506, min(p.vital_status_PRT3009506_label),
       count(DISTINCT p.vital_status_PRT3009506_label), count(*),
       max(CASE WHEN l.permissible_value_prt_id IS NULL
                 AND p.vital_status_PRT3009506 IS NOT NULL THEN 1 ELSE 0 END)
FROM partner_dev.partner_provenance.pb_demographics_person_r3_platinum p
LEFT JOIN partner_library_test.standard_library.standard_library_data_element l
  ON l.data_element_prt_id = 'PRT3009506'
 AND l.permissible_value_prt_id = p.vital_status_PRT3009506
GROUP BY p.vital_status_PRT3009506
ORDER BY 1, 2;

-- F4 (supplied by CPD entity grain "one row per distinct MRN"; verified here):
-- expect rows = distinct_mrn and null_mrn = 0. If not, Q01 must count DISTINCT MRN
-- (the W4S convention "Patients means distinct medical record numbers" applies anyway).
SELECT count(*) AS rows,
       count(DISTINCT medical_record_number_PRT4161822) AS distinct_mrn,
       sum(CASE WHEN medical_record_number_PRT4161822 IS NULL THEN 1 ELSE 0 END) AS null_mrn
FROM partner_dev.partner_provenance.pb_demographics_person_r3_platinum;

-- F5 (reference result): living female patients by code and by label. Both counts
-- must agree; a difference means the code/label companions disagree.
SELECT
  count(DISTINCT CASE WHEN administrative_gender_PRTL2140000129 = 'PRT8532'
                       AND vital_status_PRT3009506 = 'PRT45884343'
                      THEN medical_record_number_PRT4161822 END) AS by_code,
  count(DISTINCT CASE WHEN lower(administrative_gender_PRTL2140000129_label) = 'female'
                       AND lower(vital_status_PRT3009506_label) = 'alive'
                      THEN medical_record_number_PRT4161822 END) AS by_label
FROM partner_dev.partner_provenance.pb_demographics_person_r3_platinum;
