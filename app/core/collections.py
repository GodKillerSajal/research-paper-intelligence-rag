"""
Collection management: metadata about collections stored in a simple JSON file.
"""
import json
import os
from typing import List, Dict, Any

COLLECTIONS_FILE = "./data/collections.json"


def _load() -> Dict[str, Any]:
    if not os.path.exists(COLLECTIONS_FILE):
        return {}
    with open(COLLECTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(COLLECTIONS_FILE), exist_ok=True)
    with open(COLLECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_collection(name: str, description: str = "") -> Dict[str, Any]:
    data = _load()
    if name in data:
        return data[name]
    data[name] = {"name": name, "description": description, "documents": []}
    _save(data)
    return data[name]


def get_collection(name: str) -> Dict[str, Any]:
    data = _load()
    return data.get(name)


def list_collections() -> List[Dict[str, Any]]:
    data = _load()
    return list(data.values())


def delete_collection(name: str) -> bool:
    data = _load()
    if name not in data:
        return False
    del data[name]
    _save(data)
    return True


def add_document_to_collection(
    collection_name: str,
    document_info: Dict[str, Any]
) -> None:
    data = _load()
    if collection_name not in data:
        create_collection(collection_name)
        data = _load()
    # Avoid duplicates by document_id
    existing_ids = [d["document_id"] for d in data[collection_name]["documents"]]
    if document_info["document_id"] not in existing_ids:
        data[collection_name]["documents"].append(document_info)
    _save(data)


def get_documents(collection_name: str) -> List[Dict[str, Any]]:
    data = _load()
    col = data.get(collection_name, {})
    return col.get("documents", [])
