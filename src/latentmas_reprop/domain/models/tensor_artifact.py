"""Domain exception contracts for file-based experiment storage."""


class StorageError(Exception):
    """Base exception for all storage operations."""


class ConfigurationMismatchError(StorageError, ValueError):
    """Raised when resuming a run with mismatched configuration settings."""


class ManifestCorruptedError(StorageError):
    """Raised when a storage manifest file exists on disk but is invalid or corrupted."""


class TensorMissingError(StorageError, FileNotFoundError):
    """Raised when a required tensor file does not exist on disk."""


class TensorChecksumMismatchError(StorageError, ValueError):
    """Raised when a stored tensor file fails checksum integrity check."""


class ImmutableTensorOverwriteError(StorageError, FileExistsError):
    """Raised when attempting to overwrite an immutable file with conflicting content."""
