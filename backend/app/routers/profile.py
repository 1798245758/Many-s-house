from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.common import ApiResponse
from app.schemas.profile import ProfileOut, ProfileUpdate, SettingOut, SettingUpdate
from app.models.user import User
from app.models.setting import Setting

router = APIRouter(prefix="/api", tags=["profile"])

def _get_or_create_user(db: Session) -> User:
    user = db.query(User).first()
    if not user:
        user = User(nickname="用户")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

@router.get("/profile", response_model=ApiResponse)
def get_profile(db: Session = Depends(get_db)):
    user = _get_or_create_user(db)
    return ApiResponse(data=ProfileOut.model_validate(user).model_dump())

@router.put("/profile", response_model=ApiResponse)
def update_profile(req: ProfileUpdate, db: Session = Depends(get_db)):
    user = _get_or_create_user(db)
    if req.nickname is not None:
        user.nickname = req.nickname
    if req.email is not None:
        user.email = req.email
    if req.avatar is not None:
        user.avatar = req.avatar
    db.commit()
    db.refresh(user)
    return ApiResponse(code="SUCCESS", message="更新成功", data=ProfileOut.model_validate(user).model_dump())

@router.get("/settings", response_model=ApiResponse)
def get_settings(db: Session = Depends(get_db)):
    theme = db.query(Setting).filter(Setting.key == "theme").first()
    api_key = db.query(Setting).filter(Setting.key == "api_key").first()
    return ApiResponse(data=SettingOut(
        theme=theme.value if theme else "light",
        api_key_configured=bool(api_key and api_key.value),
    ).model_dump())

@router.put("/settings", response_model=ApiResponse)
def update_settings(req: SettingUpdate, db: Session = Depends(get_db)):
    if req.api_key is not None:
        s = db.query(Setting).filter(Setting.key == "api_key").first()
        if s:
            s.value = req.api_key
        else:
            db.add(Setting(key="api_key", value=req.api_key))
    if req.theme is not None:
        s = db.query(Setting).filter(Setting.key == "theme").first()
        if s:
            s.value = req.theme
        else:
            db.add(Setting(key="theme", value=req.theme))
    db.commit()
    return ApiResponse(code="SUCCESS", message="设置已更新")
