from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from mahjong_ai.common.seat import Seat
from mahjong_ai.web.schemas import (
    ActionDescriptor,
    CreateSessionRequest,
    SeatControllerConfig,
    SeatControllerKind,
)
from mahjong_ai.web.session import SessionManager


manager = SessionManager()


def _seat(value: int | str) -> Seat:
    return Seat(int(value))


def create_session_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    controllers_payload = payload.get("seat_controllers", {})
    controllers = {
        _seat(seat): SeatControllerConfig(
            SeatControllerKind(config.get("kind", "human")),
            config.get("model_id"),
            config.get("provider"),
            config.get("base_url"),
            config.get("token"),
            config.get("model_name"),
        )
        for seat, config in controllers_payload.items()
    }
    session = manager.create_session(
        CreateSessionRequest(
            rule_id=payload["rule_id"],
            seed=int(payload.get("seed", 1)),
            dealer=_seat(payload.get("dealer", 0)),
            seat_controllers=controllers,
        )
    )
    return asdict(session.summary)


try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Mahjong AI Dev Table API")
    STATIC_DIR = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/rules")
    def list_rules() -> list[dict[str, Any]]:
        return [asdict(rule) for rule in manager.list_rules()]

    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return [asdict(session) for session in manager.list_sessions()]

    @app.post("/api/sessions")
    def create_session(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return create_session_from_payload(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            return asdict(manager.get(session_id).summary)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/snapshot")
    def get_snapshot(session_id: str, viewer: int | None = None) -> dict[str, Any]:
        try:
            return manager.get(session_id).snapshot(
                viewer=Seat(viewer) if viewer is not None else None
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/legal-actions")
    def get_legal_actions(session_id: str) -> dict[str, Any]:
        try:
            return asdict(manager.get(session_id).decision_snapshot())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/actions")
    def submit_action(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            return session.step(
                ActionDescriptor.from_payload(payload),
                viewer=Seat(payload["viewer"]) if "viewer" in payload else None,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/step")
    def step(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            return session.step(
                ActionDescriptor.from_payload(payload),
                viewer=Seat(payload["viewer"]) if "viewer" in payload else None,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/controller-step")
    def controller_step(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            actor = Seat(int(payload["actor"]))
            return session.controller_step(
                actor,
                viewer=Seat(payload["viewer"]) if "viewer" in payload else None,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/pass")
    def submit_pass(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            session.submit_pass(Seat(int(payload["actor"])))
            return session.snapshot(
                viewer=Seat(payload["viewer"]) if "viewer" in payload else None
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/advance")
    def advance(session_id: str) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            session.advance_to_decision()
            return session.snapshot()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/events")
    def events(
        session_id: str,
        since: int = 0,
        viewer: int | None = None,
    ) -> list[dict[str, Any]]:
        try:
            return list(
                manager.get(session_id).events(
                    since=since,
                    viewer=Seat(viewer) if viewer is not None else None,
                )
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/replay")
    def replay(session_id: str) -> dict[str, Any]:
        try:
            return manager.get(session_id).replay()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/full-log")
    def full_log(session_id: str) -> dict[str, Any]:
        try:
            return manager.get(session_id).full_log()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

except ImportError:
    app = None
