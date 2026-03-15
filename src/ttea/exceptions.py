class TTEAError(Exception):
    """Base error type for the platform."""


class ConfigError(TTEAError):
    """Raised when a configuration file is invalid."""


class DatasetUnavailableError(TTEAError):
    """Raised when a dataset path does not contain usable files."""


class ExecutionBlockedError(TTEAError):
    """Raised when an agent cannot continue and no support path exists."""


class EnvironmentIntegrationError(TTEAError):
    """Raised when a real environment backend cannot be initialized or executed."""


class ModelBackendError(TTEAError):
    """Raised when a transformers or torch backend cannot be initialized."""


class PersistenceError(TTEAError):
    """Raised when experiment artifacts cannot be saved."""


class TrainingError(TTEAError):
    """Raised when a training pipeline cannot be built or executed."""
