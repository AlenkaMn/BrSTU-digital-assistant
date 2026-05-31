from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, BigInteger, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Mapped

Base = declarative_base()

@dataclass
class UserRaw:
    user_id: int
    num_toxic_query: int
    created_at: datetime
    updated_at: datetime

@dataclass
class MessageRaw:
    message_id: int
    user_id: int
    user_message: str
    system_answer: str
    is_toxic: bool
    is_social_query: bool
    user_mood: str
    created_at: datetime


class User(Base):
    __tablename__ = 'users'

    user_id: Mapped[int] = Column(BigInteger, primary_key=True, index=True)
    num_toxic_query: Mapped[int] = Column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    messages = relationship("Message", back_populates="user")


class Message(Base):
    __tablename__ = 'messages'

    message_id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = Column(BigInteger, ForeignKey('users.user_id'), nullable=False)
    user_message: Mapped[str] = Column(Text, nullable=False)
    system_answer: Mapped[str] = Column(Text, nullable=True)
    is_toxic: Mapped[bool] = Column(Boolean, default=False)
    is_social_query: Mapped[bool] = Column(Boolean, default=False)
    user_mood: Mapped[str] = Column(String(255), nullable=True)
    created_at: Mapped[datetime] = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="messages")


class ChatDB:
    def __init__(self, db_path: str = "sqlite:///chat_history.db"):
        self.engine = create_engine(db_path, echo=False, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def _session(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def add_user(self, user_id: int):
        with self._session() as session:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                session.add(User(user_id=user_id))

    def add_message(self, user_id: int, user_message: str, system_answer: str = "",
                    is_toxic: bool = False, is_social_query: bool = False, user_mood: str = None):
        """Сохраняет сообщение в базу."""
        self.add_user(user_id)  # убедимся, что пользователь существует
        with self._session() as session:
            session.add(
                Message(
                    user_id=user_id,
                    user_message=user_message,
                    system_answer=system_answer,
                    is_toxic=is_toxic,
                    is_social_query=is_social_query,
                    user_mood=user_mood
                )
            )

    def get_last_messages(self, user_id: int, limit: int = 5):
        """Возвращает последние N сообщений пользователя."""
        with self._session() as session:
            messages = (
                session.query(Message)
                .filter_by(user_id=user_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                MessageRaw(
                    message_id=m.message_id,
                    user_id=m.user_id,
                    user_message=m.user_message,
                    system_answer=m.system_answer,
                    is_toxic=m.is_toxic,
                    is_social_query=m.is_social_query,
                    user_mood=m.user_mood,
                    created_at=m.created_at
                )
                for m in messages
            ]


if __name__ == "__main__":
    db = ChatDB()
    db.add_message(1234, "Привет, как дела?", "Здравствуйте! Всё хорошо.", is_toxic=False, is_social_query=True)
    for msg in db.get_last_messages(1234):
        print(msg)