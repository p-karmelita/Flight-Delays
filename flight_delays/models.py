from database import Base
from sqlalchemy import Column, Float


class Flights(Base):
    __tablename__ = 'flights'

    departure_time = Column(Float)
    departure_delay = Column(Float)
    scheduled_time = Column(Float)
    arrival_time = Column(Float)
    arrival_delay = Column(Float, primary_key=True)
