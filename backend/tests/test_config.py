from app.config import BASE_DIR, DATA_DIR, UPLOAD_DIR, DB_PATH, init_dirs


def test_base_dir_points_to_backend():
    assert BASE_DIR.name == "backend"


def test_data_dir_is_under_backend():
    assert DATA_DIR.parent == BASE_DIR
    assert DATA_DIR.name == "data"


def test_db_path():
    assert DB_PATH.name == "knowledge.db"
    assert DB_PATH.parent == DATA_DIR


def test_upload_dir():
    assert UPLOAD_DIR.name == "uploads"
    assert UPLOAD_DIR.parent == BASE_DIR


def test_init_dirs_creates_directories():
    init_dirs()
    assert DATA_DIR.exists()
    assert DATA_DIR.is_dir()
    assert UPLOAD_DIR.exists()
    assert UPLOAD_DIR.is_dir()
