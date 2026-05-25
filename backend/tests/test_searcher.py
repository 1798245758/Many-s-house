from app.services.retrieval.searcher import cosine_similarity

def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 0.001

def test_cosine_similarity_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

def test_cosine_similarity_negative():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
