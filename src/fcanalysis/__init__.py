from importlib.metadata import version

from .format import ConversationSample

__version__ = version("fcanalysis")

__all__ = ["ConversationSample", "__version__"]
