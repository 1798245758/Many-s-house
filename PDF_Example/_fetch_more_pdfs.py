# -*- coding: utf-8 -*-
"""扩充金标样本：07 下载 NIST 报告（页眉页脚+页码），08 本地合成中文扫描页（OCR 金标）"""
import os, urllib.request
from pypdf import PdfReader, PdfWriter

OUT = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 07：NIST SP 800-34r1，多页、每页页眉页脚+页码（_strip_repeated_lines 金标）
p07 = os.path.join(OUT, "07_页眉页脚_NIST_SP800-34r1.pdf")
if os.path.exists(p07):
    print("07 | 已存在，跳过")
else:
    for url in ["https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-34r1.pdf",
                "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-34r1.pdf"]:
        try:
            data = urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=120).read()
            assert data[:5] == b"%PDF-", "not a pdf"
            open(p07, "wb").write(data)
            print(f"07 | {len(data)//1024}KB | {len(PdfReader(p07).pages)}页 | {url}")
            break
        except Exception as e:
            print(f"07 | {url} 失败: {e}")

# 08：中文文本渲染成图像再包成无文本层 PDF——OCR 端到端金标（内容确定可断言）
p08 = os.path.join(OUT, "08_扫描件_中文合成_员工手册节选.pdf")
if os.path.exists(p08):
    print("08 | 已存在，跳过")
else:
    from PIL import Image, ImageDraw, ImageFont
    LINES = ["胖东来员工手册节选", "第一条 员工应遵守国家法律法规。",
             "第二条 公司提供五险一金及带薪年假。", "第三条 员工应爱护公司财物。"]
    font = None
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
               r"C:\Windows\Fonts\simsun.ttc"]:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 42)
            break
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(LINES):
        d.text((120, 200 + i * 140), line, fill="black", font=font)
    img_pdf = os.path.join(OUT, "_tmp08.pdf")
    img.save(img_pdf, "PDF", resolution=150)
    # 重打包为纯图像 PDF（确保无文本层）
    w = PdfWriter()
    w.add_page(PdfReader(img_pdf).pages[0])
    with open(p08, "wb") as f:
        w.write(f)
    os.remove(img_pdf)
    print(f"08 | 合成完成 | {len(LINES)}行中文 | 无文本层")
