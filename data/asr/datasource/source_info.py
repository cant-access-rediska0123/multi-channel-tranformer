import abc


class SourceInfo(abc.ABC):
    def __init__(self, path: str):
        if path.startswith("weight="):
            weight_spec, path = path.split(":", maxsplit=1)
            _, w = weight_spec.split("=")
            self._weight = float(w)
        else:
            self._weight = 1
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    @property
    def weight(self):
        return self._weight
