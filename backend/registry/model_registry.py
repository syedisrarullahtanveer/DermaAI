"""
Semantic versioning and metadata registry for DermaAI models.
"""
class ModelRegistry:
    def __init__(self):
        self._models = {}

    def register(self, version: str, metadata: dict):
        self._models[version] = metadata

    def get_metadata(self, version: str) -> dict | None:
        return self._models.get(version)

    def list_versions(self) -> list:
        return list(self._models.keys())
