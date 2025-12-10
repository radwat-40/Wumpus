import time
import threading
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

log = logging.getLogger("MessageBus")

@dataclass
class Message:
    sender: str
    recipients: Optional[List[str]]
    topic: str
    payload: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MessageBus:
    def __init__(self, persist_file: Optional[str] = None):
        self._inboxes: Dict[str, List[Message]] = {}
        self._lock = threading.Lock()
        self.persist_file = persist_file
        if self.persist_file:
            try:
                open(self.persist_file, "a").close()
            except Exception as e:
                log.warning(f"Could not open persist file {self.persist_file}: {e}")

    def _persist(self, msg: Message):
        if not self.persist_file:
            return
        try:
            with open(self.persist_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(msg), default=str, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"Failed to persist message: {e}")

    def register(self, role: str):
        with self._lock:
            if role not in self._inboxes:
                self._inboxes[role] = []

    def send(self, sender: str, recipients: Optional[List[str]], topic: str, payload: Dict[str, Any]):
        msg = Message(sender = sender, recipients = recipients, topic = topic, payload = payload)
        with self._lock:
            if recipients is None:
                for role in list(self._inboxes.keys()):
                    self._inboxes[role].append(msg)
                log.debug(f"Broadcast from {sender}: {topic} -> payload={payload}")
            else:
                for r in recipients:
                    if r not in self._inboxes:
                        self._inboxes[r] = []
                    self._inboxes[r].append(msg)
                log.debug(f"Sent from {sender} to {recipients}: {topic} -> payload={payload}")
        self._persist(msg)

    def broadcast(self, sender: str, topic: str, payload: Dict[str, Any]):
        self.send(sender=sender, recipients=None, topic=topic, payload=payload)

    def get_messages_for(self, role: str) -> List[Message]:
        with self._lock:
            msgs = list(self._inboxes.get(role, []))
            self._inboxes[role] = []
        if msgs:
            log.debug(f"get_messages_for({role}) -> {len(msgs)} messages")
        return msgs
    
    def peef_messages_for(self, role: str) -> List[Message]:
        with self._lock:
            return list(self._inboxes.get(role, []))