"""Cluster endpoints."""

from fastapi import APIRouter, HTTPException

from cre.api.schemas import ClusterResponse
from cre.configs.clusters import get_cluster, list_clusters

router = APIRouter(prefix="/clusters", tags=["clusters"])


@router.get("/", response_model=list[str])
def get_clusters():
    return list_clusters()


@router.get("/{name}", response_model=ClusterResponse)
def get_cluster_by_name(name: str):
    try:
        cluster = get_cluster(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ClusterResponse(
        name=cluster.name,
        engine_name=cluster.engine_name,
        total_engines=cluster.total_engines,
        base_diameter=cluster.base_diameter,
        rings=[r.model_dump() for r in cluster.rings],
    )
