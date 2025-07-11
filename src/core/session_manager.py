"""
Session management for user state and temporary data.

This module handles user sessions, temporary file management,
and session cleanup with automatic timeout handling.
"""

import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class SessionManager:
    """Manages user sessions and temporary data."""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=2)
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session_dir = self.temp_dir / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions[session_id] = {
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "temp_dir": session_dir,
            "generation_history": [],
            "cached_results": {},
            "user_preferences": {}
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data if it exists and is valid."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if datetime.utcnow() - session["last_activity"] > self.session_timeout:
            self.cleanup_session(session_id)
            return None
        
        session["last_activity"] = datetime.utcnow()
        return session
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up a session and its temporary files."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            temp_dir = session.get("temp_dir")
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up session directory: {temp_dir}")
                except Exception as e:
                    logger.error(f"Failed to clean up session directory: {e}")
            
            del self.sessions[session_id]
            logger.info(f"Session cleaned up: {session_id}")
    
    def cleanup_expired_sessions(self) -> None:
        """Clean up all expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_activity"] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the generation history for a session."""
        session = self.get_session(session_id)
        if session:
            return session.get("generation_history", [])
        return []
    
    def add_to_session_history(self, session_id: str, item: Dict[str, Any]) -> None:
        """Add an item to session history."""
        session = self.get_session(session_id)
        if session:
            session["generation_history"].append(item)
            session["last_activity"] = datetime.utcnow()
    
    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences for a session."""
        session = self.get_session(session_id)
        if session:
            return session.get("user_preferences", {})
        return {}
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]) -> None:
        """Update user preferences for a session."""
        session = self.get_session(session_id)
        if session:
            session["user_preferences"].update(preferences)
            session["last_activity"] = datetime.utcnow()
    
    def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.sessions)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get basic session information."""
        session = self.get_session(session_id)
        if session:
            return {
                "session_id": session_id,
                "created_at": session["created_at"],
                "last_activity": session["last_activity"],
                "generation_count": len(session.get("generation_history", [])),
                "temp_dir": str(session["temp_dir"])
            }
        return None
