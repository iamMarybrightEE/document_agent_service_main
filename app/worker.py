from celery import Celery

from app.core.config import settings


celery_app = Celery("meeting_agent", broker=settings.redis_url, backend=settings.redis_url)


@celery_app.task(name="meeting_agent.ping")
def ping() -> str:
    return "pong"
