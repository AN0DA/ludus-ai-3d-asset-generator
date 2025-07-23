from datetime import datetime, timedelta

from src.core.session_manager import SessionManager


def test_create_session(session_manager: SessionManager) -> None:
    session_id = session_manager.create_session()
    assert session_id in session_manager.sessions, "Session should be created"
    assert (session_manager.temp_dir / "sessions" / session_id).exists(), "Session directory should exist"


def test_get_session(session_manager: SessionManager) -> None:
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    assert session is not None, "Valid session should be retrieved"
    assert "created_at" in session, "Session should have created_at"
    assert "last_activity" in session, "Session should have last_activity"


def test_session_timeout(session_manager: SessionManager) -> None:
    session_id = session_manager.create_session()
    session_manager.sessions[session_id]["last_activity"] = (
        datetime.utcnow() - session_manager.session_timeout - timedelta(seconds=1)
    )
    session = session_manager.get_session(session_id)
    assert session is None, "Expired session should return None"
    assert session_id not in session_manager.sessions, "Expired session should be removed"


def test_session_history(session_manager: SessionManager) -> None:
    session_id = session_manager.create_session()
    item = {"asset_id": "test_asset"}
    session_manager.add_to_session_history(session_id, item)
    history = session_manager.get_session_history(session_id)
    assert len(history) == 1, "Session history should contain one item"
    assert history[0] == item, "Session history item should match"


def test_cleanup_expired_sessions(session_manager: SessionManager) -> None:
    session_id = session_manager.create_session()
    session_manager.sessions[session_id]["last_activity"] = (
        datetime.utcnow() - session_manager.session_timeout - timedelta(seconds=1)
    )
    session_manager.cleanup_expired_sessions()
    assert session_id not in session_manager.sessions, "Expired sessions should be cleaned up"
