from __future__ import annotations
from datetime import date
from functools import wraps
import warnings
from sqlalchemy.exc import PendingRollbackError, InvalidRequestError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

def safe_execute(method):
    """
    Decorator for OmopKnowledgeGraph methods.

    If the session is in a failed transaction state, roll back and re-raise.
    Does NOT clear caches.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except PendingRollbackError:
            warnings.warn(
                f"Session rollback triggered in {method.__name__}",
                RuntimeWarning,
            )
            self.rollback_session()
            raise
        except InvalidRequestError as e:
            if "rollback" in str(e).lower():
                self.rollback_session()
            raise
    return wrapper


def make_engine(
    url: str,
    *,
    echo: bool = False,
    connect_timeout: int = 10,
):
    kwargs = {}
    if not url.startswith("sqlite"):
        kwargs["connect_args"] = {"connect_timeout": connect_timeout}
    return create_engine(url, echo=echo, **kwargs)


def make_session(
    url: str,
    *,
    echo: bool = False,
) -> Session:
    engine = make_engine(url, echo=echo)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()