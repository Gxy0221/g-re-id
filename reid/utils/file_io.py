import errno
import logging
import os
import shutil
from collections import OrderedDict
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Union,
)

__all__ = ["PathManager", "get_cache_dir"]


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("FVCORE_CACHE", "~/.torch/fvcore_cache")
        )
    return cache_dir


class PathHandler:
    _strict_kwargs_check = True

    def _check_kwargs(self, kwargs: Dict[str, Any]) -> None:
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError("Unused arguments: {}".format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning(
                    "[PathManager] {}={} argument ignored".format(k, v)
                )

    def _get_supported_prefixes(self) -> List[str]:
        raise NotImplementedError()

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
       
        raise NotImplementedError()

    def _open(
            self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
       
        raise NotImplementedError()

    def _copy(
            self,
            src_path: str,
            dst_path: str,
            overwrite: bool = False,
            **kwargs: Any,
    ) -> bool:
        
        raise NotImplementedError()

    def _exists(self, path: str, **kwargs: Any) -> bool:
        
        raise NotImplementedError()

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        
        raise NotImplementedError()

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        
        raise NotImplementedError()

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        
        raise NotImplementedError()

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        
        raise NotImplementedError()

    def _rm(self, path: str, **kwargs: Any) -> None:
       
        raise NotImplementedError()


class NativePathHandler(PathHandler):
    

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        self._check_kwargs(kwargs)
        return path

    def _open(
            self,
            path: str,
            mode: str = "r",
            buffering: int = -1,
            encoding: Optional[str] = None,
            errors: Optional[str] = None,
            newline: Optional[str] = None,
            closefd: bool = True,
            opener: Optional[Callable] = None,
            **kwargs: Any,
    ) -> Union[IO[str], IO[bytes]]:
       
        self._check_kwargs(kwargs)
        return open(  
            path,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    def _copy(
            self,
            src_path: str,
            dst_path: str,
            overwrite: bool = False,
            **kwargs: Any,
    ) -> bool:
      
        self._check_kwargs(kwargs)

        if os.path.exists(dst_path) and not overwrite:
            logger = logging.getLogger(__name__)
            logger.error("Destination file {} already exists.".format(dst_path))
            return False

        try:
            shutil.copyfile(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Error in file copy - {}".format(str(e)))
            return False

    def _exists(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.exists(path)

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.isfile(path)

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.isdir(path)

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        self._check_kwargs(kwargs)
        return os.listdir(path)

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        self._check_kwargs(kwargs)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def _rm(self, path: str, **kwargs: Any) -> None:
        self._check_kwargs(kwargs)
        os.remove(path)


class PathManager:
    _PATH_HANDLERS: MutableMapping[str, PathHandler] = OrderedDict()
    _NATIVE_PATH_HANDLER = NativePathHandler()

    @staticmethod
    def __get_path_handler(path: str) -> PathHandler:
        
        for p in PathManager._PATH_HANDLERS.keys():
            if path.startswith(p):
                return PathManager._PATH_HANDLERS[p]
        return PathManager._NATIVE_PATH_HANDLER

    @staticmethod
    def open(
            path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:     
        return PathManager.__get_path_handler(path)._open(  
            path, mode, buffering=buffering, **kwargs
        )

    @staticmethod
    def copy(
            src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
      
        assert PathManager.__get_path_handler(  
            src_path
        ) == PathManager.__get_path_handler(dst_path)
        return PathManager.__get_path_handler(src_path)._copy(
            src_path, dst_path, overwrite, **kwargs
        )

    @staticmethod
    def get_local_path(path: str, **kwargs: Any) -> str:
       
        return PathManager.__get_path_handler(  
            path
        )._get_local_path(path, **kwargs)

    @staticmethod
    def exists(path: str, **kwargs: Any) -> bool:
        
        return PathManager.__get_path_handler(path)._exists(  
            path, **kwargs
        )

    @staticmethod
    def isfile(path: str, **kwargs: Any) -> bool:
        
        return PathManager.__get_path_handler(path)._isfile(  
            path, **kwargs
        )

    @staticmethod
    def isdir(path: str, **kwargs: Any) -> bool:
        
        return PathManager.__get_path_handler(path)._isdir(  
            path, **kwargs
        )

    @staticmethod
    def ls(path: str, **kwargs: Any) -> List[str]:
        
        return PathManager.__get_path_handler(path)._ls(  
            path, **kwargs
        )

    @staticmethod
    def mkdirs(path: str, **kwargs: Any) -> None:
        
        return PathManager.__get_path_handler(path)._mkdirs( 
            path, **kwargs
        )

    @staticmethod
    def rm(path: str, **kwargs: Any) -> None:
        
        return PathManager.__get_path_handler(path)._rm(  
            path, **kwargs
        )

    @staticmethod
    def register_handler(handler: PathHandler) -> None:
        
        assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            assert prefix not in PathManager._PATH_HANDLERS
            PathManager._PATH_HANDLERS[prefix] = handler

        
        PathManager._PATH_HANDLERS = OrderedDict(
            sorted(
                PathManager._PATH_HANDLERS.items(),
                key=lambda t: t[0],
                reverse=True,
            )
        )

    @staticmethod
    def set_strict_kwargs_checking(enable: bool) -> None:
       
        PathManager._NATIVE_PATH_HANDLER._strict_kwargs_check = enable
        for handler in PathManager._PATH_HANDLERS.values():
            handler._strict_kwargs_check = enable
