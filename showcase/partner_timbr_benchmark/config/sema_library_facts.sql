-- Library facts export for arm S (supplied input, not intake discovery).
-- Scope: every de_id named by the two CPD v2 product specs.
-- Statement 1: one row per data element (label, definition, permissible-value count).
SELECT data_element_prt_id,
       max(data_element_prt_preferred_label) AS label,
       max(data_element_definition) AS definition,
       count(DISTINCT permissible_value_prt_id) AS pv_count
FROM partner_library_test.standard_library.standard_library_data_element
WHERE data_element_prt_id IN ('PRT3009506', 'PRT3052558', 'PRT35918286', 'PRT35918306', 'PRT35918319', 'PRT35918328', 'PRT35918383', 'PRT35918387', 'PRT35918472', 'PRT35918562', 'PRT35918597', 'PRT35918637', 'PRT35918704', 'PRT35918735', 'PRT35918746', 'PRT35918791', 'PRT35918889', 'PRT4013886', 'PRT4031696', 'PRT4053609', 'PRT4083587', 'PRT4110772', 'PRT4154129', 'PRT4155450', 'PRT4179218', 'PRT4181605', 'PRT4216292', 'PRT4216765', 'PRT4265167', 'PRT4265453', 'PRT430549', 'PRT4335813', 'PRT44803968', 'PRT914042', 'PRTL2140000129', 'PRTL2140000594', 'PRTL2140000906', 'PRTL2140001985', 'PRTL2140006801', 'PRTL2140007288', 'PRTL2140007289', 'PRTL2140007961', 'PRTL2140007963', 'PRTL2140008130', 'PRTL2140009899')
GROUP BY data_element_prt_id
ORDER BY data_element_prt_id
LIMIT 200;
-- Statement 2: permissible values of the elements whose full list fits the listing limit
-- (pv_count <= 80); larger lists stay reference-only. Page with OFFSET 0, 200, 400.
SELECT data_element_prt_id, permissible_value_prt_id, permissible_value_prt_preferred_label
FROM partner_library_test.standard_library.standard_library_data_element
WHERE data_element_prt_id IN ('PRT3009506', 'PRT3052558', 'PRT35918286', 'PRT35918306', 'PRT35918319', 'PRT35918328', 'PRT35918383', 'PRT35918387', 'PRT35918472', 'PRT35918562', 'PRT35918597', 'PRT35918637', 'PRT35918704', 'PRT35918735', 'PRT35918746', 'PRT35918791', 'PRT35918889', 'PRT4013886', 'PRT4031696', 'PRT4053609', 'PRT4083587', 'PRT4110772', 'PRT4154129', 'PRT4155450', 'PRT4179218', 'PRT4181605', 'PRT4216292', 'PRT4216765', 'PRT4265167', 'PRT4265453', 'PRT430549', 'PRT4335813', 'PRT44803968', 'PRT914042', 'PRTL2140000129', 'PRTL2140000594', 'PRTL2140000906', 'PRTL2140001985', 'PRTL2140006801', 'PRTL2140007288', 'PRTL2140007289', 'PRTL2140007961', 'PRTL2140007963', 'PRTL2140008130', 'PRTL2140009899')
  AND data_element_prt_id IN (
    SELECT data_element_prt_id
    FROM partner_library_test.standard_library.standard_library_data_element
    GROUP BY data_element_prt_id
    HAVING count(DISTINCT permissible_value_prt_id) <= 80)
ORDER BY data_element_prt_id, permissible_value_prt_id
LIMIT 200 OFFSET 0;
