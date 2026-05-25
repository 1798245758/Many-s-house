from pydantic import BaseModel

class ProfileOut(BaseModel):
    id: int
    nickname: str
    email: str | None = None
    avatar: str | None = None
    model_config = {"from_attributes": True}

class ProfileUpdate(BaseModel):
    nickname: str | None = None
    email: str | None = None
    avatar: str | None = None

class SettingOut(BaseModel):
    theme: str = "light"
    api_key_configured: bool = False
    api_key: str = ""

class SettingUpdate(BaseModel):
    api_key: str | None = None
    theme: str | None = None
