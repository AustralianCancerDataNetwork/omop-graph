# Utils for the optional omop-emb package

class MissingExtensionError(ImportError):
    """Raised when an optional omop extension is required but not installed."""
    pass