from pydantic import BaseModel, Field
from typing import List, Optional

class Experience(BaseModel):
    company: str = Field(..., min_length=1, description="Компания")
    position: str = Field(..., min_length=1, description="Должность")
    years: int = Field(..., ge=0, le=60, description="Лет опыта")
    description: str = Field(default="", description="Описание")

class Education(BaseModel):
    university: str = Field(..., min_length=1, description="Университет")
    degree: str = Field(..., min_length=1, description="Степень")
    graduation_year: int = Field(..., ge=1950, le=2100, description="Год окончания")

class Resume(BaseModel):
    name: str = Field(..., min_length=1, description="ФИО")
    email: Optional[str] = Field(default=None, description="Email")
    phone: Optional[str] = Field(default=None, description="Телефон")
    skills: List[str] = Field(default_factory=list, description="Навыки")
    experience: List[Experience] = Field(default_factory=list, description="Опыт работы")
    education: List[Education] = Field(default_factory=list, description="Образование")
