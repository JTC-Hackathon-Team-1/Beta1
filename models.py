from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, Float, Enum, Table, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime

from database import Base

# =====================================================
# Common Enums
# =====================================================

class UserRole(enum.Enum):
    admin = "admin"
    teacher = "teacher"
    student = "student"
    housing_seeker = "housing_seeker"
    property_manager = "property_manager"
    jurisdiction_admin = "jurisdiction_admin"

class EnrollmentStatus(enum.Enum):
    active = "active"
    completed = "completed"
    cancelled = "cancelled"

class ResourceType(enum.Enum):
    pdf = "pdf"
    video = "video"
    audio = "audio"
    link = "link"

class CourseLevel(enum.Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"

# =====================================================
# Association Tables
# =====================================================

# Association between User and Listing (favorite listings)
user_favorite_listings = Table(
    "_favorite_listings",
    Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("user_account_id", Integer, ForeignKey("users.id")),
    Column("listing_id", Integer, ForeignKey("listings.id"))
)

# Association between Applications and UnitTypes
application_unit_types = Table(
    "_ApplicationsToUnitTypes",
    Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("application_id", Integer, ForeignKey("applications.id")),
    Column("unit_type_id", Integer, ForeignKey("unit_types.id"))
)

# =====================================================
# CasaLingua Models
# =====================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    profile_picture = Column(String, nullable=True)
    bio = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.student, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    preferred_language = Column(String, ForeignKey("languages.code"), nullable=True)

    # CasaLingua Relationships
    courses_teaching = relationship("Course", back_populates="teacher")
    enrollments = relationship("Enrollment", back_populates="user")
    lesson_progress = relationship("LessonProgress", back_populates="user")
    submissions = relationship("Submission", back_populates="user")
    reviews = relationship("Review", back_populates="user")
    participations = relationship("Participant", back_populates="user")
    messages = relationship("Message", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    
    # Bloom Housing Relationships
    favorite_listings = relationship("Listing", secondary=user_favorite_listings)
    applications = relationship("Application", back_populates="user")
    language = relationship("Language")

class Language(Base):
    __tablename__ = "languages"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    code = Column(String, nullable=False, unique=True)
    flag_icon = Column(String, nullable=True)

    # Relationships
    courses = relationship("Course", back_populates="language")
    users = relationship("User", back_populates="language")
    translation_audits = relationship("TranslationAudit", 
                                    primaryjoin="Language.code==TranslationAudit.language")

class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    language_id = Column(Integer, ForeignKey("languages.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    level = Column(Enum(CourseLevel), nullable=False)
    price = Column(Float, nullable=False)
    is_published = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    max_students = Column(Integer, nullable=True)
    cover_image = Column(String, nullable=True)
    
    # For integration with Bloom Housing
    related_property_id = Column(Integer, ForeignKey("properties.id"), nullable=True)

    # Relationships
    teacher = relationship("User", back_populates="courses_teaching")
    language = relationship("Language", back_populates="courses")
    lessons = relationship("Lesson", back_populates="course")
    enrollments = relationship("Enrollment", back_populates="course")
    reviews = relationship("Review", back_populates="course")
    related_property = relationship("Property")

class Lesson(Base):
    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    order_index = Column(Integer, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    course = relationship("Course", back_populates="lessons")
    resources = relationship("Resource", back_populates="lesson")
    assignments = relationship("Assignment", back_populates="lesson")
    progress = relationship("LessonProgress", back_populates="lesson")

class Enrollment(Base):
    __tablename__ = "enrollments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    enrollment_date = Column(DateTime, default=func.now(), nullable=False)
    status = Column(Enum(EnrollmentStatus), default=EnrollmentStatus.active, nullable=False)
    amount_paid = Column(Float, nullable=False)

    # Relationships
    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")

class LessonProgress(Base):
    __tablename__ = "lesson_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    lesson_id = Column(Integer, ForeignKey("lessons.id"), nullable=False)
    is_completed = Column(Boolean, default=False, nullable=False)
    progress_percentage = Column(Integer, default=0, nullable=False)
    last_accessed = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="lesson_progress")
    lesson = relationship("Lesson", back_populates="progress")

class Resource(Base):
    __tablename__ = "resources"

    id = Column(Integer, primary_key=True, index=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id"), nullable=False)
    title = Column(String, nullable=False)
    type = Column(Enum(ResourceType), nullable=False)
    url = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    # Relationships
    lesson = relationship("Lesson", back_populates="resources")

class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True, index=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    due_date = Column(DateTime, nullable=False)
    points_possible = Column(Integer, nullable=False)

    # Relationships
    lesson = relationship("Lesson", back_populates="assignments")
    submissions = relationship("Submission", back_populates="assignment")

class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    submitted_at = Column(DateTime, default=func.now(), nullable=False)
    score = Column(Integer, nullable=True)
    feedback = Column(Text, nullable=True)

    # Relationships
    assignment = relationship("Assignment", back_populates="submissions")
    user = relationship("User", back_populates="submissions")

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="reviews")
    course = relationship("Course", back_populates="reviews")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    is_group = Column(Boolean, default=False, nullable=False)

    # Relationships
    participants = relationship("Participant", back_populates="conversation")
    messages = relationship("Message", back_populates="conversation")

class Participant(Base):
    __tablename__ = "participants"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    joined_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="participants")
    user = relationship("User", back_populates="participations")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=func.now(), nullable=False)
    is_read = Column(Boolean, default=False, nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User", back_populates="messages")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="notifications")

# =====================================================
# Bloom Housing Models
# =====================================================

class TranslationAudit(Base):
    __tablename__ = "translation_audits"

    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    language = Column(String, ForeignKey("languages.code"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    bias_check = Column(String)
    pii_compliance = Column(String)
    semantic_fidelity = Column(String)
    confidence_score = Column(String)
    
    # Relationships
    user = relationship("User")

class UnitType(Base):
    __tablename__ = "unit_types"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    num_bedrooms = Column(Integer, nullable=False)
    num_bathrooms = Column(Float, nullable=False)
    min_occupancy = Column(Integer, nullable=True)
    max_occupancy = Column(Integer, nullable=True)
    
    # Relationships
    units = relationship("Unit", back_populates="unit_type")
    applications = relationship("Application", secondary=application_unit_types)

class Property(Base):
    __tablename__ = "properties"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    address = Column(String, nullable=False)
    city = Column(String, nullable=False)
    state = Column(String, nullable=False)
    zip_code = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    year_built = Column(Integer, nullable=True)
    manager_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    manager = relationship("User")
    listings = relationship("Listing", back_populates="property")
    units = relationship("Unit", back_populates="property")
    associated_courses = relationship("Course")
    
class Unit(Base):
    __tablename__ = "units"
    
    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    unit_type_id = Column(Integer, ForeignKey("unit_types.id"), nullable=False)
    number = Column(String, nullable=False)
    floor = Column(Integer, nullable=True)
    square_feet = Column(Integer, nullable=True)
    accessibility_features = Column(JSON, nullable=True)
    status = Column(String, nullable=False, default="available")
    rent_amount = Column(Float, nullable=True)
    
    # Relationships
    property = relationship("Property", back_populates="units")
    unit_type = relationship("UnitType", back_populates="units")
    listing_units = relationship("ListingUnit", back_populates="unit")

class Listing(Base):
    __tablename__ = "listings"
    
    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    application_due_date = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="active")
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    published_at = Column(DateTime, nullable=True)
    application_fee = Column(Float, nullable=True)
    deposit_min = Column(Float, nullable=True)
    deposit_max = Column(Float, nullable=True)
    is_waitlist_open = Column(Boolean, default=False)
    waitlist_open_spots = Column(Integer, nullable=True)
    result_id = Column(Integer, nullable=True)
    result_link = Column(String, nullable=True)
    
    # Relationships
    property = relationship("Property", back_populates="listings")
    listing_units = relationship("ListingUnit", back_populates="listing")
    applications = relationship("Application", back_populates="listing")
    favorited_by = relationship("User", secondary=user_favorite_listings)

class ListingUnit(Base):
    __tablename__ = "listing_units"
    
    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, ForeignKey("listings.id"), nullable=False)
    unit_id = Column(Integer, ForeignKey("units.id"), nullable=False)
    rent_min = Column(Float, nullable=True)
    rent_max = Column(Float, nullable=True)
    monthly_income_min = Column(Float, nullable=True)
    annual_income_min = Column(Float, nullable=True)
    annual_income_max = Column(Float, nullable=True)
    
    # Relationships
    listing = relationship("Listing", back_populates="listing_units")
    unit = relationship("Unit", back_populates="listing_units")

class Application(Base):
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    listing_id = Column(Integer, ForeignKey("listings.id"), nullable=False)
    status = Column(String, nullable=False, default="draft")
    submitted_at = Column(DateTime, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    confirmation_code = Column(String, nullable=True, unique=True)
    
    # JSON fields for application data
    applicant_info = Column(JSON, nullable=True)
    household_members = Column(JSON, nullable=True)
    income = Column(JSON, nullable=True)
    preferences = Column(JSON, nullable=True)
    accessibility_needs = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="applications")
    listing = relationship("Listing", back_populates="applications")
    unit_types = relationship("UnitType", secondary=application_unit_types)