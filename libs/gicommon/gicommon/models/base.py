from datetime import datetime
from pydantic import BaseModel, Field
from beanie import Document, PydanticObjectId


class BaseInDB(Document):
    created_at: datetime = Field(default_factory=datetime.now)


class BaseInCode(BaseModel):
    id: PydanticObjectId
    created_at: datetime
