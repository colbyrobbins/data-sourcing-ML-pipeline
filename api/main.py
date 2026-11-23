"""Pokémon card fair-value API.

Endpoints:
    GET  /health               model and data versions
    POST /predict              card attributes (+ optional asking price) -> fair value, verdict
    GET  /search?q=...&k=5     semantic card search; results plug straight into /predict

Run locally:
    uvicorn api.main:app --reload --port 8080
Environment:
    MODEL_DIR       folder with model.joblib + model_metadata.json (default: artifacts/)
    CONFIG_PATH     config.yaml to read search settings from (default: project config.yaml)
    ENABLE_SEARCH   set to 0 to skip loading the search index and embedding model
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from pipeline.predict import DEFAULT_MODEL_DIR, load_model, predict_fair_value, price_verdict


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Card(BaseModel):
    """Raw card attributes, as shown on the card or a listing."""

    name: str = Field(examples=["Charizard ex - 199/165"])
    rarity: str = Field(examples=["Special Illustration Rare"])
    printing: str = Field("Normal", examples=["Holofoil"])
    card_type: str | None = Field(None, examples=["Fire"])
    stage: str | None = Field(None, examples=["Stage 2"])
    hp: float | None = Field(None, ge=0, examples=[330])
    retreat_cost: float | None = Field(None, ge=0, examples=[2])
    number: str | None = Field(None, examples=["199/165"])
    set_published: str | None = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Set release date, YYYY-MM-DD",
        examples=["2023-09-22"],
    )
    is_supplemental: bool = False
    has_attack2: bool = Field(False, examples=[True])


class PredictRequest(Card):
    asking_price: float | None = Field(None, gt=0, description="Optional price to judge", examples=[120])


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fair_value: float
    asking_price: float | None = None
    verdict: Literal["deal", "fair", "overpriced"] | None = None
    model_name: str
    snapshot_date: str


class SearchResult(BaseModel):
    product_id: int
    name: str
    set_name: str | None = None
    rarity: str | None = None
    printings: dict[str, float] = Field(description="Market price by printing in the snapshot")
    image_url: str | None = None
    url: str | None = None
    score: float
    card: Card = Field(description="Ready-to-send /predict body (first printing; change as needed)")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str
    snapshot_date: str
    trained_at: str
    test_metrics: dict[str, float]
    search_enabled: bool


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _default_search_loader():
    from pipeline.search import load_search

    return load_search(os.environ.get("CONFIG_PATH"))


def create_app(model_loader=None, search_loader=None) -> FastAPI:
    """Build the app. Loaders are injectable so tests can use small fakes."""
    model_loader = model_loader or (lambda: load_model(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))))
    if search_loader is None and os.environ.get("ENABLE_SEARCH", "1") != "0":
        search_loader = _default_search_loader

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.metadata = model_loader()
        app.state.search = None
        if search_loader is not None:
            try:
                app.state.search = search_loader()
            except Exception as exc:  # search is optional; /predict still works without it
                print(f"Search disabled: {exc}")
        yield

    app = FastAPI(
        title="Pokémon Card Fair-Value API",
        description="Estimate a Pokémon card's fair market price and judge an asking price.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health", response_model=Health)
    def health():
        meta = app.state.metadata
        return Health(
            status="ok",
            model_name=meta["model"],
            snapshot_date=meta["snapshot_date"],
            trained_at=meta["trained_at"],
            test_metrics=meta["test_metrics"],
            search_enabled=app.state.search is not None,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        card = req.model_dump(exclude={"asking_price"})
        fair = predict_fair_value([card], app.state.model)[0]
        meta = app.state.metadata
        return PredictResponse(
            fair_value=fair,
            asking_price=req.asking_price,
            verdict=price_verdict(req.asking_price, fair) if req.asking_price else None,
            model_name=meta["model"],
            snapshot_date=meta["snapshot_date"],
        )

    @app.get("/search", response_model=SearchResponse)
    def search(q: str = Query(..., min_length=2, max_length=200), k: int = Query(5, ge=1, le=25)):
        if app.state.search is None:
            raise HTTPException(503, "Search index is not loaded on this server.")
        results = []
        for r in app.state.search.search(q, top_k=k):
            printing = next(iter(r["printings"]), "Normal")
            card = Card(**{f: r.get(f) for f in Card.model_fields if f != "printing"}, printing=printing)
            results.append(
                SearchResult(
                    product_id=r["product_id"], name=r["name"], set_name=r.get("set_name"),
                    rarity=r.get("rarity"), printings=r["printings"], image_url=r.get("image_url"),
                    url=r.get("url"), score=r["score"], card=card,
                )
            )
        return SearchResponse(query=q, results=results)

    return app


app = create_app()
