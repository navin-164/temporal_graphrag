"""Configuration package — re-exports the settings singleton for convenient imports."""

from .settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]