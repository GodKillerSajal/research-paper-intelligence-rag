from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import CollectionCreate, CollectionResponse
from app.core.collections import (
    create_collection,
    list_collections,
    get_collection,
    delete_collection as _delete_collection,
)
from app.retrieval.vector_search import (
    delete_collection as delete_chroma_collection,
    collection_count,
)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse)
def create(body: CollectionCreate):
    col = create_collection(body.name, body.description or "")
    count = collection_count(body.name)
    return CollectionResponse(
        name=col["name"],
        description=col["description"],
        document_count=count,
    )


@router.get("", response_model=List[CollectionResponse])
def list_all():
    cols = list_collections()
    result = []
    for col in cols:
        count = collection_count(col["name"])
        result.append(CollectionResponse(
            name=col["name"],
            description=col.get("description", ""),
            document_count=count,
        ))
    return result


@router.get("/{name}", response_model=CollectionResponse)
def get(name: str):
    col = get_collection(name)
    if not col:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return CollectionResponse(
        name=col["name"],
        description=col.get("description", ""),
        document_count=collection_count(name),
    )


@router.delete("/{name}")
def delete(name: str):
    ok = _delete_collection(name)
    delete_chroma_collection(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return {"success": True, "deleted": name}
