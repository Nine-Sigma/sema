"""Errors raised by the manifest parser."""

from __future__ import annotations


class ManifestError(Exception):
    """Base class for manifest parsing errors."""


class UnsupportedManifestVersionError(ManifestError):
    pass


class UnsupportedManifestExtensionError(ManifestError):
    pass


class ManifestSchemaError(ManifestError):
    pass


class ManifestEndpointError(ManifestError):
    pass


class ManifestContextCardError(ManifestError):
    pass


class ManifestReservedNameError(ManifestError):
    pass


class ManifestVocabularyError(ManifestError):
    pass
