"""Engine endpoints."""

from fastapi import APIRouter, HTTPException

from cre.api.schemas import EngineResponse
from cre.configs.engines import get_engine, list_engines

router = APIRouter(prefix="/engines", tags=["engines"])


@router.get("/", response_model=list[str])
def get_engines():
    return list_engines()


@router.get("/{name}", response_model=EngineResponse)
def get_engine_by_name(name: str):
    try:
        engine = get_engine(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return EngineResponse(**engine.model_dump())
