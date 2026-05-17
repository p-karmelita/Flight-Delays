"""Database models."""

from sqlalchemy import Column, Float, Integer

from app.database import Base


class Flight(Base):
    """Flight model for storing flight data."""

    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    departure_time = Column(Float, nullable=False)
    departure_delay = Column(Float, nullable=False)
    scheduled_time = Column(Float, nullable=False)
    arrival_time = Column(Float, nullable=False)
    arrival_delay = Column(Float, nullable=False)

    def __repr__(self) -> str:
        """String representation of Flight."""
        return f"<Flight(id={self.id}, arrival_delay={self.arrival_delay})>"

# Made with Bob
