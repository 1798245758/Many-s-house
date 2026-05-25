import httpx

class DeepSeekClient:
    BASE_URL = "https://api.deepseek.com"
    EMBEDDING_MODEL = "deepseek-chat"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("DeepSeek API Key 未配置")
        self.api_key = api_key
        self.client = httpx.Client(timeout=60.0)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def describe_image(self, image_path: str) -> str:
        import base64, mimetypes
        mime, _ = mimetypes.guess_type(image_path)
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        resp = self.client.post(
            f"{self.BASE_URL}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": self.EMBEDDING_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": "请详细描述这张图片的内容，提取图中包含的所有文字信息和结构关系。"},
                    ],
                }],
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def embed(self, text: str) -> list[float]:
        resp = self.client.post(
            f"{self.BASE_URL}/v1/embeddings",
            headers=self._headers(),
            json={"model": self.EMBEDDING_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def chat(self, context: str, question: str) -> str:
        system_prompt = (
            "你是一个知识库助手，仅基于提供的上下文回答问题。"
            "如果上下文中没有相关信息，请诚实回答'根据现有资料无法回答此问题'。"
        )
        user_message = f"上下文:\n{context}\n\n问题: {question}"
        resp = self.client.post(
            f"{self.BASE_URL}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": self.EMBEDDING_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
