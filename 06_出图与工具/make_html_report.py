# -*- coding: utf-8 -*-
"""把客户报告.md转成自包含单文件HTML(图片base64内嵌)。"""
import base64, re, os, markdown, sys
SRC=sys.argv[1] if len(sys.argv)>1 else 'f:/1111/指纹/客户报告_指纹匹配达标实现.md'
OUT=sys.argv[2] if len(sys.argv)>2 else SRC.rsplit('.',1)[0]+'.html'
text=open(SRC,encoding='utf-8').read()
html=markdown.markdown(text,extensions=['tables'])
def embed(m):
    fn=m.group(1)
    p=os.path.join('f:/1111/指纹',fn)
    if not os.path.exists(p): return m.group(0)
    b=base64.b64encode(open(p,'rb').read()).decode()
    return f'src="data:image/png;base64,{b}"'
html=re.sub(r'src="([^"]+\.png)"',embed,html)
css="""<style>
body{font-family:'Microsoft YaHei',sans-serif;max-width:900px;margin:30px auto;padding:0 20px;line-height:1.7;color:#222}
h1{border-bottom:3px solid #2c7;padding-bottom:8px}
h2{border-bottom:1px solid #ccc;padding-bottom:5px;margin-top:34px;color:#1a6}
table{border-collapse:collapse;margin:12px 0}
th,td{border:1px solid #bbb;padding:6px 12px;text-align:center}
th{background:#eef}
img{max-width:100%;border:1px solid #ddd;margin:8px 0}
code{background:#f4f4f4;padding:2px 5px;border-radius:3px}
blockquote{border-left:4px solid #2c7;margin:10px 0;padding:6px 14px;background:#f6fff9;color:#333}
</style>"""
doc=f'<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>超声指纹匹配 达标实现报告</title>{css}</head><body>{html}</body></html>'
with open(OUT,'w',encoding='utf-8') as f: f.write(doc)
print(f'saved {OUT}  ({len(doc)//1024} KB)')
