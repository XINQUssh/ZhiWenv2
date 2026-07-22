# -*- coding: utf-8 -*-
import zipfile, re, sys, glob, os
path = r'f:\1111\指纹\(2).pptx'
z = zipfile.ZipFile(path)
slide_files = sorted([n for n in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', n)],
                     key=lambda s: int(re.findall(r'\d+', s)[-1]))
print(f"PPTX: {os.path.basename(path)}  slides={len(slide_files)}")
for sf in slide_files:
    xml = z.read(sf).decode('utf-8', 'ignore')
    texts = re.findall(r'<a:t>(.*?)</a:t>', xml, re.DOTALL)
    if not texts:
        continue
    n = re.findall(r'\d+', sf)[-1]
    print(f"\n{'='*60}\n--- Slide {n} ---")
    for t in texts:
        t = re.sub(r'<[^>]+>', '', t).strip()
        if t:
            print(t)
# also extract any notes
note_files = sorted([nm for nm in z.namelist() if re.match(r'ppt/notesSlides/notesSlide\d+\.xml$', nm)])
for nf in note_files:
    xml = z.read(nf).decode('utf-8','ignore')
    texts = [re.sub(r'<[^>]+>','',t).strip() for t in re.findall(r'<a:t>(.*?)</a:t>', xml, re.DOTALL)]
    texts = [t for t in texts if t]
    if texts:
        print(f"\n--- Notes {nf} ---")
        print(' | '.join(texts))
# list embedded images
imgs = [n for n in z.namelist() if n.startswith('ppt/media/')]
print(f"\nEmbedded media: {len(imgs)} -> {[os.path.basename(i) for i in imgs]}")
