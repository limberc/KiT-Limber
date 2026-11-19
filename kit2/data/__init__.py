"""KiT-2 data pipeline: raw downloads, point-in-time bars, leakage tests."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for IDEs and type checkers; real imports are lazy (__getattr__)
    from .crypto.download import CryptoDownloader
    from .us.download import USDownloader
    from .store import MarketStore

__all__ = ["CryptoDownloader", "USDownloader"]


def __getattr__(name: str):
    """Lazy exports: importing kit2.data (bars, features, tests) does not load the downloaders or their API
    clients, and `python -m kit2.data.us.download` runs without importing the module twice."""
    if name == "CryptoDownloader":
        from .crypto.download import CryptoDownloader
        return CryptoDownloader
    if name == "USDownloader":
        from .us.download import USDownloader
        return USDownloader
    if name == "MarketStore":
        from .store import MarketStore
        return MarketStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
