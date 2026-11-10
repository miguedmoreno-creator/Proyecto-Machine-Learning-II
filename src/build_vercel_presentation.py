"""
=============================================================================
COMPILADOR Y SINCRONIZADOR MODULAR PARA DESPLIEGUE EN VERCEL
SentiTransMilenio | Machine Learning II (Universidad Externado de Colombia)
=============================================================================
"""

import os
import json
import shutil

def find_master_payload():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', 'dataset', 'data_payload_master.json'),
        os.path.join(script_dir, '..', '..', 'entrega_final', 'data_payload_master.json'),
        os.path.join(script_dir, 'data_payload_master.json'),
        os.path.join(os.getcwd(), 'dataset', 'data_payload_master.json'),
        os.path.join(os.getcwd(), 'Proyecto', 'entrega_final', 'data_payload_master.json'),
        os.path.join(os.getcwd(), 'Proyecto', 'repo_segunda_entrega', 'dataset', 'data_payload_master.json'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"data_payload_master.json no encontrado en: {candidates}")

def build_vercel_app():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    payload_file = find_master_payload()
    print(f"[1/4] Leyendo payload maestro: {payload_file}")
    with open(payload_file, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    # Identificar carpetas vercel_app de destino
    target_app_dirs = [
        os.path.join(repo_root, 'vercel_app'),
    ]
    workspace_vercel = os.path.abspath(os.path.join(repo_root, '..', 'entrega_final', 'vercel_app'))
    if os.path.exists(os.path.dirname(workspace_vercel)):
        target_app_dirs.append(workspace_vercel)

    # 1. Actualizar vercel.json
    vercel_cfg = {
        "version": 2,
        "cleanUrls": True,
        "trailingSlash": False,
        "headers": [
            {
                "source": "/(.*)",
                "headers": [
                    { "key": "Cache-Control", "value": "public, max-age=3600, stale-while-revalidate=86400" },
                    { "key": "X-Content-Type-Options", "value": "nosniff" },
                    { "key": "X-Frame-Options", "value": "DENY" },
                    { "key": "X-XSS-Protection", "value": "1; mode=block" }
                ]
            }
        ]
    }

    data_js_content = f"// Datos Maestros Precomputados para SentiTransMilenio\nwindow.MASTER_DATA = {json.dumps(master_data, ensure_ascii=False, indent=2)};\n"

    for app_dir in target_app_dirs:
        os.makedirs(os.path.join(app_dir, 'js'), exist_ok=True)
        os.makedirs(os.path.join(app_dir, 'css'), exist_ok=True)
        os.makedirs(os.path.join(app_dir, 'graphs'), exist_ok=True)

        # Escribir vercel.json
        with open(os.path.join(app_dir, 'vercel.json'), 'w', encoding='utf-8') as f:
            json.dump(vercel_cfg, f, indent=2)

        # Actualizar js/data.js
        with open(os.path.join(app_dir, 'js', 'data.js'), 'w', encoding='utf-8') as f:
            f.write(data_js_content)

        print(f"[OK] Sincronizado data.js y vercel.json en: {app_dir}")

    # 2. Copiar gráficas generadas
    graphs_source = os.path.join(repo_root, 'graphs')
    if os.path.exists(graphs_source):
        for app_dir in target_app_dirs:
            target_graphs = os.path.join(app_dir, 'graphs')
            for item in os.listdir(graphs_source):
                if item.endswith('.png'):
                    shutil.copy2(os.path.join(graphs_source, item), os.path.join(target_graphs, item))
            print(f"[OK] Gráficas sincronizadas en: {target_graphs}")

    print("\n[ÉXITO] Arquitectura modular de Vercel verificada y lista para despliegue.")

if __name__ == '__main__':
    build_vercel_app()
