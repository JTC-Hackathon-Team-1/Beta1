import enum
from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String,
    Text, Float, Enum as SqlEnum, Table, JSON
)
from sqlalchemy.orm import relationship
from app.db import Base  # <<— corrected import

# ... rest of your enums, association tables, and model classes ...