"""
CasaLingua + Bloom Housing Integrated Platform
"""
from fastapi import FastAPI, Depends, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime, date
import models
from db import get_db, Base, engine
from app.text_pipeline import run_pipeline

# Automatically create tables if they don't exist.
# Ensure the connected DB user has permission to CREATE on schema "public".
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CasaLingua + Bloom Housing",
    description="Integrated platform for language learning and affordable housing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Pydantic Models
# ===============================

class TextInput(BaseModel):
    text: str
    source_language: str = "auto"
    target_language: str = "en"
    user_id: Optional[int] = None

class TranslationResponse(BaseModel):
    translated_text: str
    audit_id: int
    audit_report: Dict[str, Any]
    
class AuditQuery(BaseModel):
    language: Optional[str] = None
    user_id: Optional[int] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
class UserBase(BaseModel):
    email: str
    first_name: str
    last_name: str
    role: str = "student"
    preferred_language: Optional[str] = None
    
class UserCreate(UserBase):
    password: str
    
class User(UserBase):
    id: int
    profile_picture: Optional[str] = None
    bio: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    
    class Config:
        orm_mode = True

class PropertyBase(BaseModel):
    name: str
    address: str
    city: str
    state: str
    zip_code: str
    description: Optional[str] = None
    year_built: Optional[int] = None
    
class PropertyCreate(PropertyBase):
    manager_id: Optional[int] = None
    
class Property(PropertyBase):
    id: int
    manager_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True
        
class ListingBase(BaseModel):
    property_id: int
    name: str
    description: Optional[str] = None
    application_due_date: Optional[datetime] = None
    status: str = "active"
    application_fee: Optional[float] = None
    deposit_min: Optional[float] = None
    deposit_max: Optional[float] = None
    is_waitlist_open: bool = False
    waitlist_open_spots: Optional[int] = None
    
class ListingCreate(ListingBase):
    pass
    
class Listing(ListingBase):
    id: int
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None
    result_id: Optional[int] = None
    result_link: Optional[str] = None
    
    class Config:
        orm_mode = True
        
class ApplicationBase(BaseModel):
    user_id: int
    listing_id: int
    status: str = "draft"
    applicant_info: Dict[str, Any] = Field(default_factory=dict)
    household_members: List[Dict[str, Any]] = Field(default_factory=list)
    income: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    accessibility_needs: Dict[str, Any] = Field(default_factory=dict)
    
class ApplicationCreate(ApplicationBase):
    pass
    
class Application(ApplicationBase):
    id: int
    submitted_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    confirmation_code: Optional[str] = None
    
    class Config:
        orm_mode = True

class CourseBase(BaseModel):
    teacher_id: int
    language_id: int
    title: str
    description: str
    level: str
    price: float
    max_students: Optional[int] = None
    related_property_id: Optional[int] = None
    
class CourseCreate(CourseBase):
    pass
    
class Course(CourseBase):
    id: int
    is_published: bool
    created_at: datetime
    updated_at: datetime
    cover_image: Optional[str] = None
    
    class Config:
        orm_mode = True
        
# ===============================
# API Routes
# ===============================

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "CasaLingua + Bloom Housing Integrated Platform",
        "documentation": "/docs",
        "version": "1.0.0"
    }

# ===============================
# Translation & Language Routes
# ===============================

@app.post("/translate", response_model=TranslationResponse)
def run_translation(input_data: TextInput, db: Session = Depends(get_db)):
    """Translate text and audit the translation"""
    result = run_pipeline(
        input_data.text, 
        source_lang=input_data.source_language, 
        target_lang=input_data.target_language
    )
    
    audit = models.TranslationAudit(
        original_text=input_data.text,
        translated_text=result["translated_text"],
        language=input_data.target_language,
        user_id=input_data.user_id,
        bias_check=result["audit_report"]["bias_check"],
        pii_compliance=result["audit_report"]["pii_compliance"],
        semantic_fidelity=result["audit_report"]["semantic_fidelity"],
        confidence_score=str(result["audit_report"]["confidence_score"])
    )
    db.add(audit)
    db.commit()
    db.refresh(audit)
    
    return {
        "translated_text": result["translated_text"],
        "audit_id": audit.id,
        "audit_report": result["audit_report"]
    }

@app.post("/audits/query")
def query_audits(query_params: AuditQuery, db: Session = Depends(get_db)):
    """Query translation audits with filters"""
    query = db.query(models.TranslationAudit)
    
    if query_params.language:
        query = query.filter(models.TranslationAudit.language == query_params.language)
    
    if query_params.user_id:
        query = query.filter(models.TranslationAudit.user_id == query_params.user_id)
    
    if query_params.start_date:
        query = query.filter(func.date(models.TranslationAudit.timestamp) >= query_params.start_date)
    
    if query_params.end_date:
        query = query.filter(func.date(models.TranslationAudit.timestamp) <= query_params.end_date)
    
    return query.order_by(desc(models.TranslationAudit.timestamp)).all()

@app.get("/audits/{audit_id}")
def get_audit(audit_id: int = Path(..., description="The ID of the audit to retrieve"), db: Session = Depends(get_db)):
    """Get a specific translation audit by ID"""
    audit = db.query(models.TranslationAudit).filter(models.TranslationAudit.id == audit_id).first()
    if not audit:
        raise HTTPException(status_code=404, detail="Audit not found")
    return audit

@app.get("/languages")
def list_languages(db: Session = Depends(get_db)):
    """List all available languages"""
    return db.query(models.Language).all()

# ===============================
# Bloom Housing Routes
# ===============================

@app.get("/properties", response_model=List[Property])
def list_properties(
    city: Optional[str] = Query(None, description="Filter by city"),
    state: Optional[str] = Query(None, description="Filter by state"),
    db: Session = Depends(get_db)
):
    """List all properties with optional filters"""
    query = db.query(models.Property)
    
    if city:
        query = query.filter(func.lower(models.Property.city) == city.lower())
    
    if state:
        query = query.filter(func.lower(models.Property.state) == state.lower())
    
    return query.all()

@app.get("/properties/{property_id}", response_model=Property)
def get_property(property_id: int, db: Session = Depends(get_db)):
    """Get a specific property by ID"""
    property = db.query(models.Property).filter(models.Property.id == property_id).first()
    if not property:
        raise HTTPException(status_code=404, detail="Property not found")
    return property

@app.post("/properties", response_model=Property)
def create_property(property_data: PropertyCreate, db: Session = Depends(get_db)):
    """Create a new property"""
    new_property = models.Property(**property_data.dict())
    db.add(new_property)
    db.commit()
    db.refresh(new_property)
    return new_property

@app.get("/listings", response_model=List[Listing])
def list_listings(
    status: Optional[str] = Query(None, description="Filter by status (active, closed, etc.)"),
    is_waitlist_open: Optional[bool] = Query(None, description="Filter by waitlist status"),
    db: Session = Depends(get_db)
):
    """List all listings with optional filters"""
    query = db.query(models.Listing)
    
    if status:
        query = query.filter(models.Listing.status == status)
    
    if is_waitlist_open is not None:
        query = query.filter(models.Listing.is_waitlist_open == is_waitlist_open)
    
    return query.all()

@app.get("/listings/{listing_id}", response_model=Listing)
def get_listing(listing_id: int, db: Session = Depends(get_db)):
    """Get a specific listing by ID"""
    listing = db.query(models.Listing).filter(models.Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    return listing

@app.post("/listings", response_model=Listing)
def create_listing(listing_data: ListingCreate, db: Session = Depends(get_db)):
    """Create a new listing"""
    new_listing = models.Listing(**listing_data.dict())
    db.add(new_listing)
    db.commit()
    db.refresh(new_listing)
    return new_listing

@app.post("/applications", response_model=Application)
def create_application(application_data: ApplicationCreate, db: Session = Depends(get_db)):
    """Create a new housing application"""
    new_application = models.Application(**application_data.dict())
    db.add(new_application)
    db.commit()
    db.refresh(new_application)
    return new_application

@app.get("/users/{user_id}/applications", response_model=List[Application])
def get_user_applications(user_id: int, db: Session = Depends(get_db)):
    """Get all applications for a specific user"""
    applications = db.query(models.Application).filter(models.Application.user_id == user_id).all()
    return applications

@app.post("/users/{user_id}/favorites/{listing_id}")
def add_favorite_listing(user_id: int, listing_id: int, db: Session = Depends(get_db)):
    """Add a listing to a user's favorites"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    listing = db.query(models.Listing).filter(models.Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Check if already favorited
    existing = db.query(models.user_favorite_listings).filter_by(
        user_account_id=user_id, 
        listing_id=listing_id
    ).first()
    
    if existing:
        return {"message": "Listing already in favorites"}
    
    # Add to favorites
    user.favorite_listings.append(listing)
    db.commit()
    
    return {"message": "Listing added to favorites"}

@app.delete("/users/{user_id}/favorites/{listing_id}")
def remove_favorite_listing(user_id: int, listing_id: int, db: Session = Depends(get_db)):
    """Remove a listing from a user's favorites"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    listing = db.query(models.Listing).filter(models.Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Remove from favorites
    if listing in user.favorite_listings:
        user.favorite_listings.remove(listing)
        db.commit()
        return {"message": "Listing removed from favorites"}
    else:
        return {"message": "Listing not in favorites"}

# ===============================
# CasaLingua Course Routes
# ===============================

@app.get("/courses", response_model=List[Course])
def list_courses(
    language_id: Optional[int] = Query(None, description="Filter by language ID"),
    level: Optional[str] = Query(None, description="Filter by level (beginner, intermediate, advanced)"),
    related_property_id: Optional[int] = Query(None, description="Filter by related property"),
    db: Session = Depends(get_db)
):
    """List all courses with optional filters"""
    query = db.query(models.Course).filter(models.Course.is_published == True)
    
    if language_id:
        query = query.filter(models.Course.language_id == language_id)
    
    if level:
        query = query.filter(models.Course.level == level)
    
    if related_property_id:
        query = query.filter(models.Course.related_property_id == related_property_id)
    
    return query.all()

@app.get("/properties/{property_id}/courses", response_model=List[Course])
def get_property_courses(property_id: int, db: Session = Depends(get_db)):
    """Get all courses related to a specific property"""
    property = db.query(models.Property).filter(models.Property.id == property_id).first()
    if not property:
        raise HTTPException(status_code=404, detail="Property not found")
    
    courses = db.query(models.Course).filter(
        models.Course.related_property_id == property_id,
        models.Course.is_published == True
    ).all()
    
    return courses

@app.post("/courses", response_model=Course)
def create_course(course_data: CourseCreate, db: Session = Depends(get_db)):
    """Create a new course"""
    new_course = models.Course(**course_data.dict(), is_published=False)
    db.add(new_course)
    db.commit()
    db.refresh(new_course)
    return new_course

@app.put("/courses/{course_id}/publish")
def publish_course(course_id: int, db: Session = Depends(get_db)):
    """Publish a course, making it available to students"""
    course = db.query(models.Course).filter(models.Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    course.is_published = True
    course.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(course)
    
    return {"message": "Course published successfully", "course_id": course.id}

# ===============================
# Integrated Routes
# ===============================

@app.get("/properties/{property_id}/recommended-languages")
def get_property_language_recommendations(property_id: int, db: Session = Depends(get_db)):
    """Get language recommendations for a specific property based on residents"""
    property = db.query(models.Property).filter(models.Property.id == property_id).first()
    if not property:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Find all users with applications to this property's listings
    listings = db.query(models.Listing).filter(models.Listing.property_id == property_id).all()
    listing_ids = [listing.id for listing in listings]
    
    if not listing_ids:
        return {"property_id": property_id, "languages": []}
    
    applications = db.query(models.Application).filter(models.Application.listing_id.in_(listing_ids)).all()
    user_ids = [application.user_id for application in applications]
    
    # Get languages preferred by these users
    language_preferences = db.query(
        models.Language.code, 
        models.Language.name,
        func.count(models.User.id).label('user_count')
    ).join(
        models.User, 
        models.User.preferred_language == models.Language.code
    ).filter(
        models.User.id.in_(user_ids)
    ).group_by(
        models.Language.code, 
        models.Language.name
    ).order_by(
        desc('user_count')
    ).all()
    
    return {
        "property_id": property_id,
        "languages": [
            {
                "code": lang.code,
                "name": lang.name,
                "user_count": lang.user_count
            } for lang in language_preferences
        ]
    }

@app.get("/users/{user_id}/housing-language-profile")
def get_user_housing_language_profile(user_id: int, db: Session = Depends(get_db)):
    """Get integrated housing and language profile for a user"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user's housing applications
    applications = db.query(models.Application).filter(models.Application.user_id == user_id).all()
    
    # Get user's language courses
    enrollments = db.query(models.Enrollment).filter(models.Enrollment.user_id == user_id).all()
    course_ids = [enrollment.course_id for enrollment in enrollments]
    courses = db.query(models.Course).filter(models.Course.id.in_(course_ids)).all()
    
    # Get user's favorite listings
    favorite_listings = user.favorite_listings
    
    # Calculate language proficiencies based on course completions
    language_proficiencies = []
    if course_ids:
        languages = {}
        for course in courses:
            if course.language_id not in languages:
                language = db.query(models.Language).filter(models.Language.id == course.language_id).first()
                languages[course.language_id] = {
                    "id": language.id,
                    "name": language.name,
                    "code": language.code,
                    "courses_count": 0,
                    "completed_courses": 0
                }
            
            languages[course.language_id]["courses_count"] += 1
            
            # Check if course is completed
            enrollment = next((e for e in enrollments if e.course_id == course.id), None)
            if enrollment and enrollment.status == "completed":
                languages[course.language_id]["completed_courses"] += 1
        
        for lang_id, lang_data in languages.items():
            completion_percentage = 0
            if lang_data["courses_count"] > 0:
                completion_percentage = (lang_data["completed_courses"] / lang_data["courses_count"]) * 100
            
            language_proficiencies.append({
                "language_id": lang_id,
                "language_name": lang_data["name"],
                "language_code": lang_data["code"],
                "courses_enrolled": lang_data["courses_count"],
                "courses_completed": lang_data["completed_courses"],
                "completion_percentage": completion_percentage
            })
    
    return {
        "user_id": user.id,
        "name": f"{user.first_name} {user.last_name}",
        "preferred_language": user.preferred_language,
        "housing_applications": [
            {
                "id": app.id,
                "listing_id": app.listing_id,
                "status": app.status,
                "submitted_at": app.submitted_at
            } for app in applications
        ],
        "language_courses": [
            {
                "id": course.id,
                "title": course.title,
                "level": course.level,
                "language_id": course.language_id,
                "related_property_id": course.related_property_id
            } for course in courses
        ],
        "favorite_listings": [
            {
                "id": listing.id,
                "name": listing.name,
                "property_id": listing.property_id
            } for listing in favorite_listings
        ],
        "language_proficiencies": language_proficiencies
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)