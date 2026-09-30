# -*- coding: utf-8 -*-
"""下载 6 个典型 PDF 样例并校验：页数 / 可提取字符数 / 内嵌图像数"""
import os, re, urllib.request
from pypdf import PdfReader

OUT = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()

targets = [
    ("01_传统文本_规整排版_注意力论文单栏.pdf", ["https://arxiv.org/pdf/1706.03762v7", "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"]),
    ("02_多栏排版_LLM综述.pdf", ["https://arxiv.org/pdf/2303.18223"]),
    ("03_嵌入表格_加州裁员预警报告.pdf", ["https://raw.githubusercontent.com/jsvine/pdfplumber/stable/examples/pdfs/ca-warn-report.pdf"]),
    ("04_流程图与条款_NIST_SP800-34r1.pdf", ["https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-34r1.pdf", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-34r1.pdf"]),
    ("05_扫描件_纯图像无文本层_哈克贝利页29.pdf", ["https://raw.githubusercontent.com/ocrmypdf/OCRmyPDF/main/tests/resources/c03-29.pdf"]),
]
# 第 6 个：gov.uk 带签名栏的合同样本，资产直链需从页面 HTML 解析
try:
    html = get("https://www.gov.uk/government/publications/standard-crime-contract-2025").decode("utf-8", "ignore")
    cand = [u for u in re.findall(r'href="(https://assets\.publishing\.service\.gov\.uk/[^"]+\.pdf)"', html) if "signature" in u.lower()]
    if cand:
        targets.append(("06_签名栏合同_govuk犯罪合同签署样本.pdf", [cand[0]]))
    else:
        print("gov.uk 签名合同直链未解析到，跳过")
except Exception as e:
    print("gov.uk 解析失败:", e)

for name, urls in targets:
    path = os.path.join(OUT, name)
    if os.path.exists(path):  # 已下载成功的跳过
        print(f"{name} | 已存在，跳过")
        continue
    for url in urls:
        try:
            data = get(url)
            assert data[:5] == b"%PDF-", "not a pdf"
            open(path, "wb").write(data)
            r = PdfReader(path)
            chars = sum(len((p.extract_text() or "")) for p in r.pages)
            imgs = sum(len(p.images) for p in r.pages)
            print(f"{name} | {len(data)//1024}KB | {len(r.pages)}页 | 文本{chars}字 | 图像{imgs}个 | {url}")
            break
        except Exception as e:
            print(f"{name} | {url} 失败: {e}")
