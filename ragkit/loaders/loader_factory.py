"""
Purpose
-------
Select the appropriate document loader based on the source type.

Responsibilities
----------------
- Maintain the list of supported loaders.
- Select a loader that supports the supplied SourceDocument.
- Return the selected loader instance.

Does NOT
--------
- Load document contents directly.
- Parse documents.
- Chunk documents.
"""

from __future__ import annotations

from ragkit.loaders.docx_loader import DocxLoader
from ragkit.loaders.loader import Loader
from ragkit.loaders.text_loader import TextLoader
from ragkit.models.source_document import SourceDocument


class LoaderFactory:
    """
    Factory responsible for selecting the appropriate loader.
    """

    #
    # Keep all supported loaders in one place.
    #
    # LoaderFactory will check them in order and use the
    # first loader whose supports() method returns True.
    #
    _LOADERS = (
        TextLoader,
        DocxLoader,
    )

    @classmethod
    def get_loader(
        cls,
        source: SourceDocument,
    ) -> Loader:
        """
        Return the loader that supports the supplied source.

        Parameters
        ----------
        source
            Source document to be loaded.

        Returns
        -------
        Loader
            Matching loader instance.

        Raises
        ------
        LoaderError
            If no registered loader supports the source.
        """

        for loader in cls._LOADERS:

            if loader.supports(source):
                return loader()

        from ragkit.exceptions.loader_error import LoaderError

        raise LoaderError(
            f"No loader supports source: {type(source).__name__}"
        )