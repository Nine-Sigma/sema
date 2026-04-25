from __future__ import annotations

import hashlib

import pytest

from sema.ingest.naming import sanitize_schema_name


@pytest.mark.unit
class TestSanitizeSchemaName:
    def test_conforming_id_passes_through(self) -> None:
        assert sanitize_schema_name("cbioportal", "msk_chord_2024") == "cbioportal_msk_chord_2024"

    def test_lowercases_input(self) -> None:
        assert sanitize_schema_name("cbioportal", "MSK_CHORD") == "cbioportal_msk_chord"

    def test_replaces_hyphens_with_underscores(self) -> None:
        assert (
            sanitize_schema_name("cbioportal", "BRCA-TCGA-Pan-Cancer")
            == "cbioportal_brca_tcga_pan_cancer"
        )

    def test_replaces_non_identifier_chars(self) -> None:
        assert sanitize_schema_name("cbioportal", "study/v1.0") == "cbioportal_study_v1_0"

    def test_collapses_runs_of_underscores(self) -> None:
        assert sanitize_schema_name("cbioportal", "a---b___c") == "cbioportal_a_b_c"

    def test_strips_leading_and_trailing_underscores(self) -> None:
        assert sanitize_schema_name("cbioportal", "--abc--") == "cbioportal_abc"

    def test_truncates_with_hash_when_over_63(self) -> None:
        long_id = "x" * 100
        result = sanitize_schema_name("cbioportal", long_id)
        assert len(result) <= 63
        digest10 = hashlib.sha256(long_id.encode("utf-8")).hexdigest()[:10]
        assert result.endswith(f"_{digest10}")
        assert result.startswith("cbioportal_")

    def test_hash_suffix_is_deterministic(self) -> None:
        long_id = "study-" + ("z" * 100)
        first = sanitize_schema_name("cbioportal", long_id)
        second = sanitize_schema_name("cbioportal", long_id)
        assert first == second

    def test_disambiguates_long_ids_sharing_prefix(self) -> None:
        a = "x" * 80 + "_alpha"
        b = "x" * 80 + "_beta"
        result_a = sanitize_schema_name("cbioportal", a)
        result_b = sanitize_schema_name("cbioportal", b)
        assert result_a != result_b
        assert len(result_a) <= 63 and len(result_b) <= 63

    def test_short_collisions_not_resolved_by_hash(self) -> None:
        # Hash suffix only kicks in when truncation applies.
        # Short IDs that sanitize identically MUST collide so registry layer can fail fast.
        first = sanitize_schema_name("cbioportal", "BRCA-TCGA")
        second = sanitize_schema_name("cbioportal", "BRCA_TCGA")
        assert first == second == "cbioportal_brca_tcga"

    def test_result_never_exceeds_63_chars(self) -> None:
        for n in (50, 63, 64, 80, 200, 1000):
            result = sanitize_schema_name("cbioportal", "x" * n)
            assert len(result) <= 63, f"length {len(result)} for n={n}"

    def test_truncation_does_not_leave_trailing_underscore_before_hash(self) -> None:
        # Construct an ID where the natural truncation point lands on an underscore.
        # cbioportal_ prefix = 11 chars, _<hash10> suffix = 11 chars, leaves 41 for sanitized.
        long_id = "x" * 40 + "_" + "y" * 50
        result = sanitize_schema_name("cbioportal", long_id)
        digest10 = hashlib.sha256(long_id.encode("utf-8")).hexdigest()[:10]
        assert result.endswith(f"_{digest10}")
        assert "__" not in result

    def test_raises_when_prefix_leaves_no_room(self) -> None:
        with pytest.raises(ValueError):
            sanitize_schema_name("a" * 60, "x" * 100)

    def test_empty_study_id_raises(self) -> None:
        with pytest.raises(ValueError):
            sanitize_schema_name("cbioportal", "")

    def test_study_id_only_specials_raises(self) -> None:
        with pytest.raises(ValueError):
            sanitize_schema_name("cbioportal", "---")
