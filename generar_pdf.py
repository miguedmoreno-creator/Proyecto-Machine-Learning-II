"""
Convertidor Markdown → HTML estilizado → PDF (mediante Edge headless)
Segunda Entrega — SentiTransMilenio — Machine Learning II
"""

import os
import re
import subprocess
import sys

# Instalar markdown si no está disponible
try:
    import markdown
    from markdown.extensions.tables import TableExtension
    from markdown.extensions.toc import TocExtension
    from markdown.extensions.fenced_code import FencedCodeExtension
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'markdown'], check=True)
    import markdown
    from markdown.extensions.tables import TableExtension
    from markdown.extensions.toc import TocExtension
    from markdown.extensions.fenced_code import FencedCodeExtension

# Leer el markdown
md_path = 'Segunda_Entrega.md'
with open(md_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convertir a HTML
md_parser = markdown.Markdown(extensions=[
    TableExtension(),
    TocExtension(toc_depth='2-3'),
    FencedCodeExtension(),
    'nl2br',
    'sane_lists',
    'attr_list',
])
body_html = md_parser.convert(md_content)

# Reemplazar rutas relativas de imágenes con absolutas para Edge
md_dir = os.path.abspath('.')
body_html = body_html.replace('src="graphs/', f'src="file:///{md_dir}/graphs/'.replace('\\', '/'))

# ─────────────────────────────────────────────
# CSS PROFESIONAL PARA INFORME ACADÉMICO
# ─────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Merriweather:ital,wght@0,400;0,700;1,400&family=Fira+Code:wght@400;500&display=swap');

:root {
  --primary: #1a365d;
  --secondary: #2A9D8F;
  --accent: #E63946;
  --boost: #E76F51;
  --neutral: #f7f9fc;
  --border: #dce3ed;
  --text: #1f2937;
  --muted: #6b7280;
  --link: #1a56db;
  --code-bg: #f0f4f8;
  --shadow: 0 1px 4px rgba(0,0,0,0.08);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 10.5pt;
  color: var(--text);
  background: #ffffff;
  line-height: 1.72;
  max-width: 860px;
  margin: 0 auto;
  padding: 40px 48px 60px 48px;
}

/* ── Portada / Cabecera ── */
.cover {
  text-align: center;
  padding: 24px 0 12px;
  border-bottom: 3px solid var(--primary);
  margin-bottom: 36px;
}
.cover .logo {
  width: 72px; height: 72px;
  background: var(--primary);
  border-radius: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 18px;
  box-shadow: 0 4px 16px rgba(26,54,93,0.2);
}
.cover .logo svg { fill: white; width: 40px; height: 40px; }
.cover h1 {
  font-family: 'Merriweather', Georgia, serif;
  font-size: 19pt;
  font-weight: 700;
  color: var(--primary);
  letter-spacing: -0.02em;
  line-height: 1.3;
  max-width: 680px;
  margin: 0 auto 12px;
}
.cover .subtitle {
  font-size: 10pt;
  color: var(--muted);
  font-style: italic;
  margin-bottom: 8px;
}
.cover .badges {
  display: flex;
  gap: 8px;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: 14px;
}
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 8pt;
  font-weight: 600;
  letter-spacing: 0.03em;
}
.badge-blue { background: #dbeafe; color: #1e40af; }
.badge-green { background: #d1fae5; color: #065f46; }
.badge-red { background: #fee2e2; color: #991b1b; }

/* ── Headings ── */
h1 {
  font-family: 'Merriweather', serif;
  font-size: 18pt; color: var(--primary);
  margin: 40px 0 14px;
  padding-bottom: 8px;
  border-bottom: 2.5px solid var(--primary);
  page-break-after: avoid;
}
h2 {
  font-family: 'Inter', sans-serif;
  font-size: 13pt; font-weight: 700;
  color: var(--primary);
  margin: 30px 0 10px;
  padding-left: 12px;
  border-left: 4px solid var(--secondary);
  page-break-after: avoid;
}
h3 {
  font-family: 'Inter', sans-serif;
  font-size: 11pt; font-weight: 600;
  color: #374151;
  margin: 22px 0 8px;
  page-break-after: avoid;
}
h4 { font-size: 10.5pt; font-weight: 600; color: #4b5563; margin: 16px 0 6px; }

/* ── Párrafos y texto ── */
p { margin: 0 0 10px; orphans: 3; widows: 3; }
strong { font-weight: 600; color: var(--primary); }
em { font-style: italic; }

/* ── Bloques de citación / resumen ── */
blockquote {
  background: var(--neutral);
  border-left: 4px solid var(--secondary);
  margin: 16px 0;
  padding: 12px 18px;
  border-radius: 0 8px 8px 0;
  font-size: 10pt;
  color: #374151;
}
blockquote strong { color: var(--secondary); }

/* ── Tablas ── */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0 22px;
  font-size: 9pt;
  page-break-inside: avoid;
  box-shadow: var(--shadow);
  border-radius: 8px;
  overflow: hidden;
}
thead tr {
  background: var(--primary);
  color: white;
}
thead th {
  padding: 9px 12px;
  text-align: left;
  font-weight: 600;
  font-size: 8.5pt;
  letter-spacing: 0.02em;
  white-space: nowrap;
}
tbody tr { border-bottom: 1px solid var(--border); }
tbody tr:nth-child(even) { background: var(--neutral); }
tbody tr:last-child { border-bottom: none; }
tbody td {
  padding: 8px 12px;
  vertical-align: top;
  line-height: 1.4;
}
tbody td:first-child { font-weight: 500; }

/* ── Código ── */
code {
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 8.5pt;
  background: var(--code-bg);
  padding: 2px 6px;
  border-radius: 4px;
  color: #be185d;
}
pre {
  background: #0f172a;
  color: #e2e8f0;
  padding: 16px 18px;
  border-radius: 10px;
  overflow-x: auto;
  margin: 14px 0;
  page-break-inside: avoid;
  font-size: 8.5pt;
  line-height: 1.5;
}
pre code { background: none; color: inherit; padding: 0; }

/* ── Listas ── */
ul, ol { margin: 8px 0 12px 20px; }
li { margin-bottom: 4px; }
li p { margin: 0; }

/* ── Imágenes ── */
img {
  max-width: 100%;
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  display: block;
  margin: 16px auto;
  page-break-inside: avoid;
}
em:has(img) + em, img + em, p > em:last-child {
  display: block;
  text-align: center;
  font-size: 8.5pt;
  color: var(--muted);
  margin-top: -6px;
  margin-bottom: 16px;
  font-style: italic;
}

/* ── Separadores ── */
hr {
  border: none;
  border-top: 1.5px solid var(--border);
  margin: 30px 0;
}

/* ── Callout de matemáticas ── */
.math-block {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  padding: 10px 16px;
  margin: 12px 0;
  font-family: 'Fira Code', monospace;
  font-size: 9pt;
}

/* ── Mermaid / diagramas (fallback texto) ── */
.mermaid, .language-mermaid {
  background: var(--neutral);
  border: 1px dashed var(--border);
  border-radius: 8px;
  padding: 12px;
  font-size: 8pt;
  color: var(--muted);
  font-family: 'Fira Code', monospace;
}

/* ── Print / PDF ── */
@page {
  size: A4;
  margin: 20mm 18mm 22mm 18mm;
}
@media print {
  body > h1:first-child { margin-top: 0; }
  a { color: inherit; text-decoration: none; }
}

/* ── Resaltado especial para tablas de resultados ── */
table tbody td:nth-child(3),
table tbody td:nth-child(4) { font-weight: 500; }
"""

# ─────────────────────────────────────────────
# HTML completo
# ─────────────────────────────────────────────
html_document = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title></title>
  <style>{css}</style>
</head>
<body>

  <!-- CUERPO DEL MARKDOWN CONVERTIDO -->
  {body_html}

</body>
</html>
"""

# Guardar HTML
html_path = os.path.abspath('Segunda_Entrega.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_document)

print(f"HTML generado correctamente: {html_path}")

# ─────────────────────────────────────────────
# Convertir a PDF usando Edge headless
# ─────────────────────────────────────────────
edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
pdf_path = os.path.abspath('Segunda_Entrega.pdf')
html_url = f"file:///{html_path.replace(os.sep, '/')}"

cmd = [
    edge_exe,
    '--headless=new',
    '--disable-gpu',
    '--disable-extensions',
    '--no-sandbox',
    '--disable-web-security',
    f'--print-to-pdf={pdf_path}',
    '--print-to-pdf-no-header',
    '--no-pdf-header-footer',
    '--run-all-compositor-stages-before-draw',
    '--virtual-time-budget=10000',
    html_url
]

print(f"\nConvirtiendo a PDF con Edge headless...")
print(f"Destino: {pdf_path}")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"PDF generado exitosamente: {pdf_path} ({size_kb:.1f} KB)")
else:
    print(f"Error al generar PDF: {result.stderr}")
    print("El HTML está disponible para conversión manual.")
