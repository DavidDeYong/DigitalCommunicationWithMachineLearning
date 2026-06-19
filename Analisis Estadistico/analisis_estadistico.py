"""
analisis_estadistico_ieee.py
============================
Versión IEEE-ready del análisis estadístico benchmark ML para 16-QAM.

Cambios respecto a la versión original:
  ─ Tipografía: serif (Times New Roman / Latin Modern) para máxima
    compatibilidad con el estilo IEEE.
  ─ Tamaños de fuente: calibrados para doble columna IEEE (~88 mm).
    Si la figura ocupa columna completa (~180 mm) usar figsize_1col = False.
  ─ Exportación dual: PNG 600 dpi (revisión online) + PDF vectorial
    (cámara editorial, incrustación en LaTeX).
  ─ Grilla minimalista: solo eje Y, líneas muy suaves.
  ─ Leyendas compactas sin borde o con borde fino.
  ─ Barras de error con capsize reducido y linewidth fino.
  ─ Marcadores más grandes y líneas más gruesas para resistir reducción.
  ─ Márgenes tight_layout con pad reducido.

Fuente de datos: idéntica al script original (JSON o TXT fallback).
"""

import os, re, glob, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

try:
    from adjustText import adjust_text as _adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN IEEE
# ─────────────────────────────────────────────────────────────────────────────

# ── Tipografía ──────────────────────────────────────────────────────────────
# Intentar Times New Roman primero; si no está disponible, caer en DejaVu Serif.
# Para LaTeX puro (pdflatex), descomentar USE_LATEX = True (requiere LaTeX instalado).
USE_LATEX = False   # True → usa pdflatex (máxima calidad, más lento)

if USE_LATEX:
    matplotlib.rcParams.update({
        "text.usetex":        True,
        "font.family":        "serif",
        "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
    })
else:
    # Fuente sin LaTeX: Times New Roman si está instalada, si no DejaVu Serif
    import matplotlib.font_manager as fm
    _tnr_available = any(
        "times new roman" in f.name.lower() for f in fm.fontManager.ttflist
    )
    _serif_family = "Times New Roman" if _tnr_available else "DejaVu Serif"
    matplotlib.rcParams.update({
        "text.usetex":  False,
        "font.family":  "serif",
        "font.serif":   [_serif_family, "DejaVu Serif", "Palatino", "serif"],
    })

# ── Tamaños de fuente — columna doble IEEE (88 mm ≈ 3.46 in) ────────────────
# Regla práctica IEEE: texto base ≥ 8 pt después de reducción a tamaño final.
# A 600 dpi con figsize ≈ 3.5 in ancho, 8 pt base es adecuado.
_FS_BASE   = 8     # texto de ejes, tick labels
_FS_TITLE  = 9     # título del panel
_FS_LEGEND = 7     # leyenda
_FS_ANNOT  = 7     # anotaciones (heatmap, texto inline)

matplotlib.rcParams.update({
    "font.size":          _FS_BASE,
    "axes.titlesize":     _FS_TITLE,
    "axes.labelsize":     _FS_BASE,
    "xtick.labelsize":    _FS_BASE,
    "ytick.labelsize":    _FS_BASE,
    "legend.fontsize":    _FS_LEGEND,
    "figure.dpi":         150,       # pantalla
    "savefig.dpi":        600,       # salida final
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    # Líneas y marcadores
    "lines.linewidth":    1.5,
    "lines.markersize":   5,
    # Ejes
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    # Grid
    "grid.linewidth":     0.5,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    # Leyenda
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "legend.borderpad":   0.4,
    "legend.handlelength": 1.5,
    # Ticks
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.minor.size":   1.5,
    "ytick.minor.size":   1.5,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    # Layout
    "figure.constrained_layout.use": False,
})

# ── Tamaños de figura ────────────────────────────────────────────────────────
# Una columna IEEE ≈ 3.45 in | doble columna ≈ 7.16 in | alto libre
_W1  = 3.45   # ancho 1 columna
_W2  = 7.16   # ancho 2 columnas (full-page width)
_H1  = 2.6    # alto estándar panel único
_H2  = 2.2    # alto para subplots apilados

# Helpers
def _fig1col(h=_H1):
    """Figura de una columna."""
    return plt.subplots(figsize=(_W1, h))

def _fig2col(h=_H1):
    """Figura de dos columnas."""
    return plt.subplots(figsize=(_W2, h))

def _subplots_2col(rows, cols, h=_H1, **kw):
    return plt.subplots(rows, cols, figsize=(_W2, h), **kw)


# ─────────────────────────────────────────────────────────────────────────────
# PALETA Y MARCADORES
# ─────────────────────────────────────────────────────────────────────────────

# Colores optimizados para impresión en escala de grises + pantalla color.
# Se priorizan tonos distinguibles tanto en color como en luminancia.
COLORES = {
    "Bayes (ML)":    "#000000",   # negro
    "Logistic Reg.": "#B7950B",   # dorado oscuro  — grupo 1: equiv. estadístico
    "SVM RBF":       "#C0392B",   # rojo oscuro     — grupo 2: degradación moderada
    "SVM Lineal":    "#E67E22",   # naranja
    "SGD-Nystroem":  "#5D4037",   # marrón
    "XGBoost":       "#117A65",   # verde azulado
    "KNN":           "#2980B9",   # azul
    "MLP Simple":    "#7D3C98",   # púrpura          — grupo 3: degradación sistemática
    "Red Profunda":  "#1A8599",   # cian oscuro
    "ELM":           "#1E8449",   # verde oscuro
}

# Marcadores variados para distinguir curvas sin depender solo del color
MARKERS = {
    "Bayes (ML)":    "D",    # diamante
    "Logistic Reg.": "h",    # hexágono
    "SVM RBF":       "s",    # cuadrado
    "SVM Lineal":    "^",    # triángulo arriba
    "SGD-Nystroem":  "P",    # plus grueso
    "XGBoost":       "d",    # diamante delgado
    "KNN":           "*",    # estrella
    "MLP Simple":    "v",    # triángulo abajo
    "Red Profunda":  "o",    # círculo
    "ELM":           "X",    # X gruesa
}

# Orden lógico: Bayes primero, luego por grupo de desempeño esperado
ORDEN_CLF = list(COLORES.keys())

NOMBRES_CORTOS = {
    "Bayes (ML)":    "Bayes",
    "Logistic Reg.": "Log.\nReg.",
    "SVM RBF":       "SVM\nRBF",
    "SVM Lineal":    "SVM\nLineal",
    "SGD-Nystroem":  "SGD-\nNystr.",
    "XGBoost":       "XGB",
    "KNN":           "KNN",
    "MLP Simple":    "MLP\nSimple",
    "Red Profunda":  "Red\nProf.",
    "ELM":           "ELM",
    "Random Forest": "Rand.\nForest",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE RUTAS (igual que el original)
# ─────────────────────────────────────────────────────────────────────────────

CARPETA          = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROYECTO = os.path.dirname(CARPETA)

def _encontrar_json() -> str:
    import glob as _glob
    candidatos = [
        os.path.join(CARPETA, "resultados", "resultados_benchmark.json"),
        os.path.join(os.path.dirname(CARPETA), "resultados", "resultados_benchmark.json"),
        os.path.join(os.path.dirname(os.path.dirname(CARPETA)), "resultados", "resultados_benchmark.json"),
        os.path.join(CARPETA, "resultados_benchmark.json"),
        os.path.join(os.path.dirname(CARPETA), "resultados_benchmark.json"),
    ]
    for ruta in candidatos:
        if os.path.exists(ruta):
            return ruta
    patron_glob = os.path.join(os.path.dirname(CARPETA), "**", "resultados_benchmark.json")
    encontrados = _glob.glob(patron_glob, recursive=True)
    if encontrados:
        return encontrados[0]
    print(f"  [WARN] JSON no encontrado. Rutas probadas:")
    for r in candidatos:
        print(f"         {r}")
    return candidatos[1]

RUTA_JSON  = _encontrar_json()
DIR_SALIDA = os.path.join(CARPETA, "figuras_ieee")
PATRON_TXT = "ResultadoCompleto*seed*.txt"
SEEDS_EXCLUIR  = {0}
CLFS_EXCLUIR   = {"Random Forest", "XGBoost"}   # KNN incluido; XGBoost excluido (degradación severa a 12 dB)
# Cuántos archivos seed_*.json más recientes usar (None = todos)
N_ULTIMOS_JSON = 11
EB_N0_FORZADOS = None
EB_N0_ESTABLES = [6, 8, 10, 12]
EB_N0_ESPERADOS = [0, 2, 4, 6, 8, 10, 12, 14]

# ── Opciones de exportación ──────────────────────────────────────────────────
GUARDAR_PDF = True   # True → genera también PDF vectorial (recomendado para IEEE)
GUARDAR_PNG = True   # True → genera PNG 600 dpi


def _guardar(fig, nombre_base: str, dir_out: str):
    """Guarda la figura en PNG y/o PDF según configuración."""
    os.makedirs(dir_out, exist_ok=True)
    rutas = []
    if GUARDAR_PNG:
        r = os.path.join(dir_out, nombre_base + ".png")
        fig.savefig(r, dpi=600, bbox_inches="tight", pad_inches=0.02)
        rutas.append(r)
    if GUARDAR_PDF:
        r = os.path.join(dir_out, nombre_base + ".pdf")
        fig.savefig(r, bbox_inches="tight", pad_inches=0.02)
        rutas.append(r)
    plt.close(fig)
    for r in rutas:
        print(f"  [OK] {os.path.basename(r)}")


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS (idéntica al original)
# ─────────────────────────────────────────────────────────────────────────────

def _normalizar_nombre_clf(nombre: str) -> str:
    nombre = re.sub(r"KNN \(k=\d+\)", "KNN", nombre)
    return nombre.strip()


def cargar_desde_json(ruta_json: str, seeds_excluir: set = None) -> pd.DataFrame:
    if seeds_excluir is None:
        seeds_excluir = set()
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    corridas = datos.get("corridas", [])
    if not corridas:
        raise ValueError(f"El JSON '{ruta_json}' no contiene corridas.")
    registros = []
    seeds_cargados, seeds_omitidos = [], []
    for corrida in corridas:
        seed       = corrida.get("seed", -1)
        corrida_id = corrida.get("corrida_id", -1)
        fecha      = corrida.get("fecha", "?")
        if seed in seeds_excluir:
            seeds_omitidos.append(seed)
            continue
        seeds_cargados.append(seed)
        n_clfs = 0
        for nombre_clf, reg in corrida.get("registros", {}).items():
            clf = _normalizar_nombre_clf(nombre_clf)
            eb_n0_list    = reg.get("Eb_N0_dB",          [])
            ber_list      = reg.get("ber",                [])
            acc_list      = reg.get("accuracy",           [])
            t_train_list  = reg.get("t_entrenamiento_s",  [])
            t_inf_list    = reg.get("t_inferencia_us",    [])
            confiable_list= reg.get("confiable",          [])
            errores_list  = reg.get("n_errores",          [])
            # FLOPs: escalares por clf (independientes de Eb/N0)
            _ft_raw = reg.get("flops_train", None)
            _fi_raw = reg.get("flops_inf",   None)
            # Fallback: leer listas por Eb/N0 y tomar mediana de valores no nulos
            if _fi_raw is None:
                _fi_list = reg.get("flops_inferencia", None)
                if _fi_list:
                    _vals = [v for v in _fi_list if v is not None]
                    _fi_raw = float(np.nanmedian(_vals)) if _vals else None
            if _ft_raw is None:
                _ft_list = reg.get("flops_entrenamiento", None)
                if _ft_list:
                    _vals = [v for v in _ft_list if v is not None]
                    _ft_raw = float(np.nanmedian(_vals)) if _vals else None
            flops_train_val = float(_ft_raw) if _ft_raw is not None else np.nan
            flops_inf_val   = float(_fi_raw) if _fi_raw is not None else np.nan
            n = len(eb_n0_list)
            for i in range(n):
                registros.append({
                    "seed":        seed,
                    "corrida_id":  corrida_id,
                    "fecha":       fecha,
                    "clf":         clf,
                    "eb_n0":       int(eb_n0_list[i]),
                    "ber":         float(ber_list[i])       if i < len(ber_list)       else np.nan,
                    "accuracy":    float(acc_list[i])       if i < len(acc_list)       else np.nan,
                    "t_train":     float(t_train_list[i])   if i < len(t_train_list)   else np.nan,
                    "t_inf":       float(t_inf_list[i])     if i < len(t_inf_list)     else np.nan,
                    "confiable":   bool(confiable_list[i])  if i < len(confiable_list) else False,
                    "n_errores":   int(errores_list[i])     if i < len(errores_list)   else 0,
                    "flops_train": flops_train_val,
                    "flops_inf":   flops_inf_val,
                })
            n_clfs += 1
        print(f"  seed={seed:>6}  corrida_id={corrida_id}  "
              f"fecha={fecha}  clasificadores={n_clfs}")
    if seeds_omitidos:
        print(f"\n  [Info] Seeds omitidos (SEEDS_EXCLUIR): {seeds_omitidos}")
    if not registros:
        raise ValueError("No se cargaron registros.")
    return pd.DataFrame(registros)


_RE_BLOQUE = re.compile(
    r'->\s+(.+?)\r?\n'
    r'(?:.*?\r?\n)*?'
    r'\s+t_entrenamiento\s*=\s*([\d.]+)\s*s\r?\n'
    r'\s+t_inferencia_1\s*=\s*([\d.]+)\s*us\r?\n'
    r'\s+BER\s*=\s*([\d.e+\-]+)\s*\|.*?'
    r'Accuracy\s*=\s*([\d.]+)%\s*\|\s*(\S+)',
    re.DOTALL
)
_RE_NOMBRE  = re.compile(r'Completo(\d+)seed(\d+)', re.IGNORECASE)
_EB_N0_DEFAULT = [0, 2, 4, 6, 8, 10, 12, 14]


def _parsear_txt(ruta: str) -> list:
    nombre   = os.path.basename(ruta)
    m_nombre = _RE_NOMBRE.search(nombre)
    if not m_nombre:
        print(f"  [WARN] No se pudo extraer seed de '{nombre}' — omitido")
        return []
    corrida = int(m_nombre.group(1))
    seed    = int(m_nombre.group(2))
    txt     = open(ruta, encoding="utf-8", errors="ignore").read()
    idx     = txt.find("PASO 2")
    txt     = txt[idx:] if idx != -1 else txt
    matches = _RE_BLOQUE.findall(txt)
    from collections import defaultdict
    por_clf = defaultdict(list)
    for m in matches:
        por_clf[_normalizar_nombre_clf(m[0])].append(m)
    registros = []
    for clf, bloques in por_clf.items():
        for i, (_, t_train, t_inf, ber, acc, flag) in enumerate(bloques):
            if i >= len(_EB_N0_DEFAULT):
                break
            registros.append({
                "seed": seed, "corrida_id": corrida, "fecha": "?",
                "clf": clf, "eb_n0": _EB_N0_DEFAULT[i],
                "ber": float(ber), "accuracy": float(acc) / 100.0,
                "t_train": float(t_train), "t_inf": float(t_inf),
                "confiable": flag.strip().upper() == "OK", "n_errores": 0,
            })
    return registros


def cargar_desde_txt(carpeta: str, patron: str) -> pd.DataFrame:
    archivos = sorted(glob.glob(os.path.join(carpeta, patron)))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron TXT con patrón '{patron}'")
    todos = []
    for ruta in archivos:
        regs = _parsear_txt(ruta)
        print(f"  {os.path.basename(ruta):50s} → {len(regs)} registros")
        todos.extend(regs)
    return pd.DataFrame(todos)


def _cargar_desde_multiples_json(carpeta_resultados: str,
                                  n_ultimos: int | None = None,
                                  seeds_excluir: set | None = None) -> pd.DataFrame:
    """Carga los N archivos resultados_seed_*.json más recientes de carpeta_resultados."""
    patron = os.path.join(carpeta_resultados, "resultados_seed_*.json")
    archivos = sorted(glob.glob(patron), key=os.path.getmtime)
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos resultados_seed_*.json en {carpeta_resultados}")
    if n_ultimos is not None:
        archivos = archivos[-n_ultimos:]
    print(f"  Cargando {len(archivos)} archivo(s) seed:")
    todos = []
    for ruta in archivos:
        print(f"    {os.path.basename(ruta)}")
        df_seed = cargar_desde_json(ruta, seeds_excluir=seeds_excluir)
        todos.append(df_seed)
    return pd.concat(todos, ignore_index=True)


def cargar_datos() -> tuple[pd.DataFrame, list]:
    print("\n[Carga] Buscando fuente de datos...")
    carpeta_resultados = os.path.join(CARPETA_PROYECTO, "resultados")
    archivos_seed = glob.glob(os.path.join(carpeta_resultados, "resultados_seed_*.json"))
    if os.path.exists(RUTA_JSON):
        print(f"  Fuente: JSON unico -> {RUTA_JSON}")
        df = cargar_desde_json(RUTA_JSON, seeds_excluir=SEEDS_EXCLUIR)
        fuente = "JSON"
    elif archivos_seed:
        print(f"  Fuente: archivos seed (N_ULTIMOS_JSON={N_ULTIMOS_JSON})")
        df = _cargar_desde_multiples_json(carpeta_resultados,
                                          n_ultimos=N_ULTIMOS_JSON,
                                          seeds_excluir=SEEDS_EXCLUIR)
        fuente = "JSON-seeds"
    else:
        print(f"  JSON no encontrado. Usando TXT.")
        df = cargar_desde_txt(CARPETA, PATRON_TXT)
        fuente = "TXT"
    if CLFS_EXCLUIR:
        df = df[~df["clf"].isin(CLFS_EXCLUIR)]
    if EB_N0_FORZADOS is not None:
        eb_n0_validos = sorted(EB_N0_FORZADOS)
    else:
        seeds = df["seed"].unique()
        sets_ebn0 = [set(df[df["seed"] == s]["eb_n0"].unique()) for s in seeds]
        eb_n0_validos = sorted(set.intersection(*sets_ebn0)) if sets_ebn0 else []
        df = df[df["eb_n0"].isin(eb_n0_validos)]
    n_corridas = df["seed"].nunique()
    print(f"\n[Carga] Fuente: {fuente}")
    print(f"        Corridas válidas: {n_corridas}  seeds={sorted(df['seed'].unique())}")
    print(f"        Eb/N0 analizados: {eb_n0_validos}")
    if n_corridas < 3:
        print("  [WARN] Con < 3 corridas el IC 95% es poco fiable.")
    return df, eb_n0_validos


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES ESTADÍSTICAS (idénticas al original)
# ─────────────────────────────────────────────────────────────────────────────

def ic95(valores: np.ndarray):
    n = len(valores)
    if n < 2:
        return (valores[0], valores[0])
    m   = np.mean(valores)
    sem = stats.sem(valores)
    h   = sem * stats.t.ppf(0.975, df=n - 1)
    return (m - h, m + h)


def estadisticas_grupo(df_grupo: pd.DataFrame, col: str) -> dict:
    v = df_grupo[col].dropna().values
    if len(v) == 0:
        return {}
    lo, hi = ic95(v)
    return {
        "media":   np.mean(v),
        "mediana": np.median(v),
        "std":     np.std(v, ddof=1) if len(v) > 1 else 0,
        "cv":      np.std(v, ddof=1) / np.mean(v) * 100 if np.mean(v) > 0 and len(v) > 1 else 0,
        "ic_lo":   lo, "ic_hi":   hi, "n": len(v),
        "min":     np.min(v), "max": np.max(v),
    }


def _mediana_robusta(valores, k=1.5):
    """
    Mediana tras eliminar outliers superiores por cerca IQR.
    Solo elimina outliers por arriba (hibernación/suspensión infla t, no reduce).
    Retorna (mediana_filtrada, n_original, n_filtrado).
    """
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, 0, 0
    q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
    iqr = q3 - q1
    limite = q3 + k * iqr if iqr > 0 else np.inf
    v_filt = v[v <= limite]
    if len(v_filt) == 0:
        v_filt = v   # si todos son outliers, devolver sin filtrar
    return float(np.median(v_filt)), len(v), len(v_filt)


def ordenar_clasificadores_por_performance(df, eb_ref=10):
    sub = df[(df["eb_n0"] == eb_ref) & df["confiable"]]
    ranking = (
        sub.groupby("clf")["ber"].mean().sort_values().index.tolist()
    )
    return ranking


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1 — BER vs Eb/N0 (degradación dB + ratio) — 2 paneles
# ─────────────────────────────────────────────────────────────────────────────

def _ber_teorica_16qam(eb_n0_db: np.ndarray) -> np.ndarray:
    # Ec. (2) del artículo: Pb = (3/8) erfc(sqrt(2*Eb/(5*N0)))
    # Equivalencia: (3/4)*(1/2)*erfc(sqrt((4/5)*x)/sqrt(2))
    #             = (3/8)*erfc(sqrt(4x/10)) = (3/8)*erfc(sqrt(2x/5))  ✓
    from scipy.special import erfc
    eb_n0_lin = 10 ** (eb_n0_db / 10)
    return 0.75 * 0.5 * erfc(np.sqrt(0.8 * eb_n0_lin) / np.sqrt(2))


def fig_ber_vs_ebn0(df: pd.DataFrame, dir_out: str):
    EB_N0_PLOT     = [eb for eb in EB_N0_ESPERADOS if eb <= 12]
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]

    fig, (ax_db, ax_ratio) = plt.subplots(1, 2, figsize=(_W2, _H1 + 0.4))

    for clf in clfs_presentes:
        color  = COLORES.get(clf, "gray")
        marker = MARKERS.get(clf, "o")
        delta_db_vals, ratio_vals = [], []
        ic_db_lo, ic_db_hi = [], []
        ic_rt_lo, ic_rt_hi = [], []
        ebs_validos = []

        for eb in EB_N0_PLOT:
            sub = df[(df["clf"] == clf) & (df["eb_n0"] == eb) & df["confiable"]]
            if len(sub) == 0:
                continue
            ber_teo = _ber_teorica_16qam(np.array([float(eb)]))[0]
            if ber_teo <= 0:
                continue
            bers   = sub["ber"].values
            deltas = 10 * np.log10(np.maximum(bers, 1e-10) / ber_teo)
            ratios = bers / ber_teo
            est_d  = estadisticas_grupo(pd.DataFrame({"v": deltas}), "v")
            est_r  = estadisticas_grupo(pd.DataFrame({"v": ratios}), "v")
            delta_db_vals.append(est_d["media"])
            ic_db_lo.append(est_d["ic_lo"])
            ic_db_hi.append(est_d["ic_hi"])
            ratio_vals.append(est_r["media"])
            ic_rt_lo.append(max(est_r["ic_lo"], 0))
            ic_rt_hi.append(est_r["ic_hi"])
            ebs_validos.append(eb)

        if not ebs_validos:
            continue
        ebs = np.array(ebs_validos)

        ax_db.plot(ebs, delta_db_vals, color=color, marker=marker,
                   ms=4, lw=1.4, label=clf, zorder=3)
        ax_db.fill_between(ebs, ic_db_lo, ic_db_hi, color=color, alpha=0.12, zorder=2)

        ax_ratio.plot(ebs, ratio_vals, color=color, marker=marker,
                      ms=4, lw=1.4, label=clf, zorder=3)
        ax_ratio.fill_between(ebs, ic_rt_lo, ic_rt_hi, color=color, alpha=0.12, zorder=2)

    # ── Panel A ──
    ax_db.axhline(0, color="#777777", lw=0.9, ls="--", alpha=0.8, label="Teórica (0 dB)")
    ax_db.axhline(1, color="#aaaaaa", lw=0.6, ls=":", alpha=0.7)
    ax_db.text(11.6, 1.1, "+1 dB", fontsize=_FS_ANNOT, color="#888888")
    ax_db.set_xlabel("$E_b/N_0$ (dB)")
    ax_db.set_ylabel("Degradación vs. teórica (dB)")
    ax_db.set_title(r"(A)  $\Delta = 10\,\log_{10}(\mathrm{BER}_{clf}/\mathrm{BER}_{teo})$")
    ax_db.set_xticks(EB_N0_PLOT)
    ax_db.set_xlim(-0.5, 12.5)
    ax_db.yaxis.grid(True)
    ax_db.legend(loc="upper right", ncol=2, fontsize=_FS_LEGEND - 1)

    # ── Panel B ──
    ax_ratio.axhline(1.0, color="#777777", lw=0.9, ls="--", alpha=0.8, label=r"Teórica ($\times$1)")
    ax_ratio.axhline(2.0, color="#aaaaaa", lw=0.6, ls=":", alpha=0.7)
    ax_ratio.text(11.6, 2.1, r"$\times$2", fontsize=_FS_ANNOT, color="#888888")
    ax_ratio.set_xlabel("$E_b/N_0$ (dB)")
    ax_ratio.set_ylabel(r"Ratio  $\mathrm{BER}_{clf} / \mathrm{BER}_{teo}$")
    ax_ratio.set_title(r"(B)  Ratio respecto a la curva teórica")
    ax_ratio.set_xticks(EB_N0_PLOT)
    ax_ratio.set_xlim(-0.5, 12.5)
    ax_ratio.set_yscale("log")
    ax_ratio.yaxis.grid(True, which="both")
    ax_ratio.legend(loc="upper right", ncol=2, fontsize=_FS_LEGEND - 1)

    fig.tight_layout(pad=0.5, w_pad=0.8)
    _guardar(fig, "fig1_desviacion_vs_teorica", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2 — Boxplot BER en Eb/N0 clave
# ─────────────────────────────────────────────────────────────────────────────

def fig_boxplot_ber(df: pd.DataFrame, dir_out: str):
    ebs_clave      = [6, 8, 10, 12]
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    colores_lista  = [COLORES.get(c, "gray") for c in clfs_presentes]

    fig, axes = _subplots_2col(1, len(ebs_clave), h=_H1, sharey=False)

    for ax, eb in zip(axes, ebs_clave):
        sub = df[(df["eb_n0"] == eb) & df["confiable"] & df["clf"].isin(clfs_presentes)]
        data = [sub[sub["clf"] == clf]["ber"].values for clf in clfs_presentes]

        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5),
                        whiskerprops=dict(lw=0.8),
                        capprops=dict(lw=0.8),
                        boxprops=dict(lw=0.8),
                        flierprops=dict(marker=".", markersize=3, alpha=0.5))
        for patch, color in zip(bp["boxes"], colores_lista):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        ax.set_yscale("log")
        ax.set_title(f"$E_b/N_0 = {eb}$ dB")
        ax.set_xticks(range(1, len(clfs_presentes) + 1))
        ax.set_xticklabels(
            [NOMBRES_CORTOS.get(c, c) for c in clfs_presentes],
            fontsize=_FS_BASE - 1, rotation=0, ha="center")
        ax.set_ylabel("BER" if eb == ebs_clave[0] else "")
        ax.yaxis.grid(True, which="both")

    fig.suptitle("Distribución de BER por clasificador — $E_b/N_0$ clave",
                 fontsize=_FS_TITLE, y=1.01)
    fig.tight_layout(pad=0.4, w_pad=0.5)
    _guardar(fig, "fig2_boxplot_ber", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 3 — Degradación en dB respecto a Bayes con IC 95%
# ─────────────────────────────────────────────────────────────────────────────

def fig_degradacion_db(df: pd.DataFrame, dir_out: str):
    ebs_analisis   = [8, 10, 12]
    clfs_sin_bayes = [c for c in ORDEN_CLF if c in df["clf"].unique() and c != "Bayes (ML)"]

    fig, axes = _subplots_2col(1, len(ebs_analisis), h=_H1)

    for ax, eb in zip(axes, ebs_analisis):
        bayes_por_seed = df[(df["clf"] == "Bayes (ML)") & (df["eb_n0"] == eb) & df["confiable"]]
        if len(bayes_por_seed) == 0:
            continue
        bayes_media = bayes_por_seed["ber"].mean()
        medias_db, ic_lo_db, ic_hi_db, colores_barra = [], [], [], []

        for clf in clfs_sin_bayes:
            sub = df[(df["clf"] == clf) & (df["eb_n0"] == eb) & df["confiable"]]
            if len(sub) == 0:
                medias_db.append(np.nan); ic_lo_db.append(np.nan)
                ic_hi_db.append(np.nan); colores_barra.append("gray")
                continue
            deltas = []
            for seed in sub["seed"].unique():
                ber_clf   = sub[sub["seed"] == seed]["ber"].values
                ber_bayes = bayes_por_seed[bayes_por_seed["seed"] == seed]["ber"].values
                if len(ber_clf) > 0 and len(ber_bayes) > 0:
                    deltas.append(10 * np.log10(ber_clf[0] / ber_bayes[0]))
            if len(deltas) == 0:
                deltas = [10 * np.log10(sub["ber"].mean() / bayes_media)]
            est = estadisticas_grupo(pd.DataFrame({"ber": deltas}), "ber")
            medias_db.append(est["media"])
            ic_lo_db.append(est["media"] - est["ic_lo"])
            ic_hi_db.append(est["ic_hi"] - est["media"])
            colores_barra.append(COLORES.get(clf, "gray"))

        x = np.arange(len(clfs_sin_bayes))
        ax.bar(x, medias_db, color=colores_barra, alpha=0.75,
               edgecolor="white", linewidth=0.4, zorder=2)
        ax.errorbar(x, medias_db,
                    yerr=[ic_lo_db, ic_hi_db],
                    fmt="none", color="black", capsize=2.5, lw=0.9, zorder=5)
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"$E_b/N_0 = {eb}$ dB")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [NOMBRES_CORTOS.get(c, c) for c in clfs_sin_bayes],
            fontsize=_FS_BASE - 1, rotation=0, ha="center")
        ax.set_ylabel("Degradación vs. Bayes (dB)" if eb == ebs_analisis[0] else "")
        ax.yaxis.grid(True)

    fig.suptitle("Degradación en dB respecto al óptimo de Bayes — Media $\\pm$ IC 95\\%",
                 fontsize=_FS_TITLE, y=1.01)
    fig.tight_layout(pad=0.4, w_pad=0.5)
    _guardar(fig, "fig3_degradacion_db", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 4 — Heatmap p-values t de Student pareado (diferencias en dB)
# ─────────────────────────────────────────────────────────────────────────────

def test_pareado(df: pd.DataFrame, clf_a: str, clf_b: str, eb: int = 10) -> dict:
    """
    Compara clf_a vs clf_b en eb dB mediante dos tests sobre Delta_dB pareado.

    Delta_dB_i = 10*log10(BER_a[seed_i] / BER_b[seed_i])

    Returns dict con:
      p_ttest   : p-value t de Student (H0: media(Delta_dB) = 0)
      p_wilcoxon: p-value Wilcoxon signed-rank (no paramétrico, mismo H0)
      n_pares   : número de seeds comunes
      delta_media : media de Delta_dB (dB)
      delta_ic95  : semiancho IC95% de Delta_dB
    """
    from scipy.stats import wilcoxon as _wilcoxon
    sub   = df[(df["eb_n0"] == eb) & df["confiable"]]
    diffs = _diferencias_db_pareadas(sub, clf_a, clf_b)
    result = {"n_pares": len(diffs), "delta_media": np.nan,
              "delta_ic95": np.nan, "p_ttest": np.nan, "p_wilcoxon": np.nan}
    if len(diffs) < 2:
        return result
    result["delta_media"] = float(np.mean(diffs))
    lo, hi = ic95(diffs)
    result["delta_ic95"] = float((hi - lo) / 2)
    try:
        _, result["p_ttest"] = stats.ttest_1samp(diffs, popmean=0.0)
    except Exception:
        pass
    try:
        _, result["p_wilcoxon"] = _wilcoxon(diffs)
    except Exception:
        pass
    return result


def _diferencias_db_pareadas(sub: pd.DataFrame, clf_a: str, clf_b: str) -> np.ndarray:
    """
    Diferencias pareadas por seed: Δ_dB = 10·log10(BER_a / BER_b).
    Opera en escala logarítmica para mejorar la simetría de la distribución.
    Solo incluye seeds presentes en ambos clasificadores.
    """
    seeds_a = set(sub[sub["clf"] == clf_a]["seed"].values)
    seeds_b = set(sub[sub["clf"] == clf_b]["seed"].values)
    seeds   = sorted(seeds_a & seeds_b)
    diffs = []
    for s in seeds:
        ba = sub[(sub["clf"] == clf_a) & (sub["seed"] == s)]["ber"].values
        bb = sub[(sub["clf"] == clf_b) & (sub["seed"] == s)]["ber"].values
        if len(ba) > 0 and len(bb) > 0 and bb[0] > 0 and ba[0] > 0:
            diffs.append(10 * np.log10(ba[0] / bb[0]))
    return np.array(diffs)


def fig_heatmap_pvalues(df: pd.DataFrame, dir_out: str):
    """
    Heatmap de p-values del t de Student pareado sobre diferencias en dB.
    Coherente con el IC 95% basado en t de Student declarado en el artículo.
    Las diferencias log(BER_a/BER_b) son más simétricas que las BER crudas.
    """
    eb_ref         = 10
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    n              = len(clfs_presentes)
    sub            = df[(df["eb_n0"] == eb_ref) & df["confiable"]]
    p_matrix       = np.ones((n, n))

    for i, clf_a in enumerate(clfs_presentes):
        for j, clf_b in enumerate(clfs_presentes):
            if i == j:
                continue
            diffs = _diferencias_db_pareadas(sub, clf_a, clf_b)
            if len(diffs) < 2:
                p_matrix[i, j] = np.nan
                continue
            try:
                _, p = stats.ttest_1samp(diffs, popmean=0.0)
            except Exception:
                p = 1.0
            p_matrix[i, j] = p

    mask = np.eye(n, dtype=bool)
    p_df = pd.DataFrame(p_matrix, index=clfs_presentes, columns=clfs_presentes)

    fig, ax = plt.subplots(figsize=(_W1 + 0.5, _W1 + 0.5))
    cmap    = sns.diverging_palette(10, 145, s=80, l=55, as_cmap=True)
    sns.heatmap(p_df, annot=True, fmt=".3f", cmap=cmap,
                vmin=0, vmax=0.1, mask=mask,
                linewidths=0.4, ax=ax,
                annot_kws={"size": _FS_ANNOT - 1},
                cbar_kws={"label": "$p$-value ($t$ pareado, $\\Delta$dB)", "shrink": 0.8})

    ax.set_title(f"$p$-values $t$ de Student pareado — $\\Delta$dB @ {eb_ref} dB",
                 fontsize=_FS_TITLE)
    ax.tick_params(axis="x", rotation=35, labelsize=_FS_BASE - 1)
    ax.tick_params(axis="y", rotation=0,  labelsize=_FS_BASE - 1)

    from matplotlib.patches import Patch
    patches = [
        Patch(color=cmap(0.0), label="$p < 0.05$ (significativo)"),
        Patch(color=cmap(0.9), label="$p \\geq 0.05$"),
    ]
    ax.legend(handles=patches, loc="lower right",
              fontsize=_FS_LEGEND, framealpha=0.9)

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig4_heatmap_pvalues", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 5 — Boxplot t_train
# ─────────────────────────────────────────────────────────────────────────────

def fig_boxplot_ttrain(df: pd.DataFrame, dir_out: str):
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    colores_lista  = [COLORES.get(c, "gray") for c in clfs_presentes]
    sub            = df[df["eb_n0"].isin(EB_N0_ESTABLES)]

    fig, ax = plt.subplots(figsize=(_W2 * 0.75, _H1))
    data    = [sub[sub["clf"] == clf]["t_train"].values for clf in clfs_presentes]
    data_f  = [v[v > 0] for v in data]

    bp = ax.boxplot(data_f, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=0.7),
                    capprops=dict(lw=0.7),
                    boxprops=dict(lw=0.7),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colores_lista):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(clfs_presentes) + 1))
    ax.set_xticklabels(
        [NOMBRES_CORTOS.get(c, c) for c in clfs_presentes],
        fontsize=_FS_BASE - 1, rotation=0, ha="center")
    ax.set_ylabel("Training time (s)")
    ax.set_title(f"$t_{{train}}$ distribution — $E_b/N_0 \\in {{{EB_N0_ESTABLES}}}$ dB")
    ax.yaxis.grid(True, which="both")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig5_boxplot_ttrain", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 6 — t_train vs Eb/N0
# ─────────────────────────────────────────────────────────────────────────────

def fig_ttrain_vs_ebn0(df: pd.DataFrame, dir_out: str):
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()
                      and c != "Bayes (ML)"]

    fig, ax = plt.subplots(figsize=(_W1 + 0.6, _H1))

    for clf in clfs_presentes:
        color  = COLORES.get(clf, "gray")
        marker = MARKERS.get(clf, "o")
        medianas, q25_list, q75_list, ebs = [], [], [], []
        for eb in EB_N0_ESPERADOS:
            sub = df[(df["clf"] == clf) & (df["eb_n0"] == eb)]
            v   = sub["t_train"].values
            if len(v) == 0:
                continue
            medianas.append(np.median(v))
            q25_list.append(np.percentile(v, 25))
            q75_list.append(np.percentile(v, 75))
            ebs.append(eb)
        ebs      = np.array(ebs)
        medianas = np.array(medianas)
        ax.semilogy(ebs, medianas, color=color, marker=marker,
                    ms=4, lw=1.3, label=clf)
        ax.fill_between(ebs,
                        np.maximum(q25_list, 1e-3),
                        np.maximum(q75_list, 1e-3),
                        color=color, alpha=0.10)

    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("$t_{train}$ (s)")
    ax.set_title("Training time vs $E_b/N_0$ — Median $\\pm$ IQR")
    ax.set_xticks(EB_N0_ESPERADOS)
    ax.yaxis.grid(True, which="both")
    ax.legend(loc="upper right", ncol=2, fontsize=_FS_LEGEND)

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig6_ttrain_vs_ebn0", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 7 — Boxplot t_inf
# ─────────────────────────────────────────────────────────────────────────────

def fig_boxplot_tinf(df: pd.DataFrame, dir_out: str):
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    colores_lista  = [COLORES.get(c, "gray") for c in clfs_presentes]
    sub            = df[df["eb_n0"].isin(EB_N0_ESTABLES)]

    fig, ax = plt.subplots(figsize=(_W2 * 0.75, _H1))
    data    = [sub[sub["clf"] == clf]["t_inf"].values for clf in clfs_presentes]

    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=0.7),
                    capprops=dict(lw=0.7),
                    boxprops=dict(lw=0.7),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colores_lista):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(clfs_presentes) + 1))
    ax.set_xticklabels(
        [NOMBRES_CORTOS.get(c, c) for c in clfs_presentes],
        fontsize=_FS_BASE - 1, rotation=0, ha="center")
    ax.set_ylabel("Inference time — 1 symbol (µs)")
    ax.set_title(f"$t_{{inf}}$ distribution — $E_b/N_0 \\in {{{EB_N0_ESTABLES}}}$ dB")
    ax.yaxis.grid(True, which="both")

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig7_boxplot_tinf", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 8 — Scatter BER vs t_train
# ─────────────────────────────────────────────────────────────────────────────

def fig_scatter_ber_ttrain(df: pd.DataFrame, dir_out: str):
    eb_ref          = 10
    clfs_ordenados  = ordenar_clasificadores_por_performance(df, eb_ref)
    sub             = df[(df["eb_n0"] == eb_ref) & df["confiable"]]

    # Figura de una columna para maximizar resolución de puntos
    fig, ax = plt.subplots(figsize=(_W1 + 0.4, _H1 + 0.2))

    for clf in clfs_ordenados:
        d = sub[sub["clf"] == clf]
        if len(d) == 0:
            continue
        ber_media, _, _   = _mediana_robusta(d["ber"].values)
        train_media, _, _ = _mediana_robusta(d["t_train"].values)
        ber_media         = d["ber"].mean()
        ber_std           = d["ber"].std()
        train_iqr         = d["t_train"].quantile(0.75) - d["t_train"].quantile(0.25)

        ax.errorbar(train_media, ber_media,
                    xerr=train_iqr / 2, yerr=ber_std,
                    fmt=MARKERS.get(clf, "o"),
                    color=COLORES.get(clf, "gray"),
                    ms=5, capsize=2.5, lw=1.0,
                    label=clf, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("$t_{train}$ (s)")
    ax.set_ylabel("BER @ 10 dB")
    ax.set_title("BER vs Training Cost Trade-off")
    ax.yaxis.grid(True, which="both")
    ax.xaxis.grid(True, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=_FS_LEGEND, frameon=True)

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig8_scatter_ber_vs_ttrain", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 9 — Scatter BER vs t_inf
# ─────────────────────────────────────────────────────────────────────────────

def fig_scatter_ber_tinf(df: pd.DataFrame, dir_out: str):
    eb_ref         = 10
    clfs_ordenados = ordenar_clasificadores_por_performance(df, eb_ref)
    sub            = df[(df["eb_n0"] == eb_ref) & df["confiable"]]

    fig, ax = plt.subplots(figsize=(_W1 + 0.4, _H1 + 0.2))

    for clf in clfs_ordenados:
        d = sub[sub["clf"] == clf]
        if len(d) == 0:
            continue
        ber_media         = d["ber"].mean()
        inf_media, _, _   = _mediana_robusta(d["t_inf"].values)
        ber_std           = d["ber"].std()
        inf_iqr           = d["t_inf"].quantile(0.75) - d["t_inf"].quantile(0.25)

        ax.errorbar(inf_media, ber_media,
                    xerr=inf_iqr / 2, yerr=ber_std,
                    fmt=MARKERS.get(clf, "o"),
                    color=COLORES.get(clf, "gray"),
                    ms=5, capsize=2.5, lw=1.0,
                    label=("MLP Profundo" if clf == "Red Profunda" else clf),
                    zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("$t_{inf}$ (µs/symbol)")
    ax.set_ylabel("BER @ 10 dB")
    ax.set_title("BER vs Inference Latency Trade-off")
    ax.yaxis.grid(True, which="both")
    ax.xaxis.grid(True, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=_FS_LEGEND, frameon=True)

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig9_scatter_ber_vs_tinf", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 10 — Coeficiente de variación (CV%)
# ─────────────────────────────────────────────────────────────────────────────

def fig_cv(df: pd.DataFrame, dir_out: str):
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    sub_estable    = df[df["eb_n0"].isin(EB_N0_ESTABLES) & df["confiable"]]
    cv_ber, cv_train = [], []

    for clf in clfs_presentes:
        d = sub_estable[sub_estable["clf"] == clf]
        cv_ber.append(estadisticas_grupo(d, "ber").get("cv", 0))
        t = d["t_train"].values
        t = t[t > 0]
        cv_train.append(np.std(t, ddof=1) / np.mean(t) * 100 if len(t) > 1 else 0)

    x     = np.arange(len(clfs_presentes))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(_W2, _H1))
    ax.bar(x - ancho / 2, cv_ber,   ancho,
           label="CV\\% BER",
           color=[COLORES.get(c, "gray") for c in clfs_presentes],
           alpha=0.82, edgecolor="white", linewidth=0.4)
    ax.bar(x + ancho / 2, cv_train, ancho,
           label="CV\\% $t_{train}$",
           color=[COLORES.get(c, "gray") for c in clfs_presentes],
           alpha=0.40, edgecolor="white", linewidth=0.4, hatch="///")

    ax.axhline(5,  color="#2ECC71", ls="--", lw=0.9, alpha=0.8, label="5\\% ref. BER")
    ax.axhline(20, color="#E67E22", ls="--", lw=0.9, alpha=0.8, label="20\\% ref. $t_{train}$")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [NOMBRES_CORTOS.get(c, c) for c in clfs_presentes],
        fontsize=_FS_BASE - 1, rotation=0, ha="center")
    ax.set_ylabel("Coefficient of Variation (\\%)")
    ax.set_title(
        f"Consistency — CV\\% of BER and $t_{{train}}$ — "
        f"$E_b/N_0 \\in {{{EB_N0_ESTABLES}}}$ dB"
    )
    ax.legend(fontsize=_FS_LEGEND, framealpha=0.9)
    ax.yaxis.grid(True)

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig10_cv_consistencia", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# TABLA RESUMEN
# ─────────────────────────────────────────────────────────────────────────────

def imprimir_tabla_resumen(df: pd.DataFrame):
    n_corridas     = df["seed"].nunique()
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    sub            = df[df["eb_n0"].isin(EB_N0_ESTABLES) & df["confiable"]]
    con_flops      = _tiene_flops(df)

    ancho = 116 if con_flops else 90
    print("\n" + "=" * ancho)
    print(f"  TABLA RESUMEN — Media +- IC95% — Eb/N0 estables — {n_corridas} corridas")
    print("=" * ancho)
    header = (f"  {'Clasificador':<20}  {'BER media':>12}  {'+-IC95%':>10}  "
              f"{'CV_BER%':>8}  {'t_train(s)':>12}  {'t_inf(us)':>12}  {'n':>4}")
    if con_flops:
        header += f"  {'f_train(med)':>13}  {'f_inf(med)':>12}"
    print(header)
    print("  " + "-" * (ancho - 2))

    for clf in clfs_presentes:
        d  = sub[sub["clf"] == clf]
        eb = estadisticas_grupo(d, "ber")
        if not eb:
            continue
        tt_med, tt_n, tt_nf = _mediana_robusta(d["t_train"].values)
        ti_med, ti_n, ti_nf = _mediana_robusta(d["t_inf"].values)
        tt_flag = f"*{tt_n-tt_nf}" if tt_n > tt_nf else ""
        ti_flag = f"*{ti_n-ti_nf}" if ti_n > ti_nf else ""
        ic_ancho = (eb["ic_hi"] - eb["ic_lo"]) / 2
        row = (f"  {clf:<20}  {eb['media']:>12.4e}  {ic_ancho:>10.2e}  "
               f"{eb['cv']:>8.1f}  {tt_med:>9.2f}{tt_flag:<3}  "
               f"{ti_med:>9.1f}{ti_flag:<3}  {eb['n']:>4}")
        if con_flops:
            ft = d["flops_train"].dropna()
            fi = d["flops_inf"].dropna()
            sft = f"{ft.median():.3e}" if len(ft) > 0 else "N/A"
            sfi = f"{fi.median():.3e}" if len(fi) > 0 else "N/A"
            row += f"  {sft:>13}  {sfi:>12}"
        print(row)
    print("=" * ancho)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA B — Tabla FLOPs mediana ± IQR por clasificador
# ─────────────────────────────────────────────────────────────────────────────

def _tiene_flops(df: pd.DataFrame) -> bool:
    """True si al menos una fila tiene flops_train o flops_inf no NaN."""
    if "flops_train" not in df.columns and "flops_inf" not in df.columns:
        return False
    return (df["flops_train"].notna().any() or df["flops_inf"].notna().any())


def imprimir_tabla_flops(df: pd.DataFrame):
    if not _tiene_flops(df):
        print("\n[FLOPs] Campo flops_train/flops_inf ausente en los JSON — tabla omitida.")
        print("        TODO: agregar medición de FLOPs al pipeline principal (main.py).")
        return

    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    # Usar un Eb/N0 estable como referencia; los FLOPs son iguales para todos los Eb/N0
    sub = df[df["eb_n0"].isin(EB_N0_ESTABLES)].drop_duplicates(subset=["seed", "clf"])

    print("\n" + "=" * 80)
    print("  TABLA FLOPs — Mediana ± IQR por clasificador (Tabla V artículo)")
    print("=" * 80)
    print(f"  {'Clasificador':<20}  {'flops_train (med)':>18}  {'IQR_train':>12}"
          f"  {'flops_inf (med)':>16}  {'IQR_inf':>10}")
    print("  " + "-" * 76)

    for clf in clfs_presentes:
        d = sub[sub["clf"] == clf]
        ft = d["flops_train"].dropna().values
        fi = d["flops_inf"].dropna().values

        def _med_iqr(v):
            if len(v) == 0:
                return "N/A", "N/A"
            med = np.median(v)
            iqr = np.percentile(v, 75) - np.percentile(v, 25)
            return f"{med:.3e}", f"{iqr:.2e}"

        med_t, iqr_t = _med_iqr(ft)
        med_i, iqr_i = _med_iqr(fi)
        print(f"  {clf:<20}  {med_t:>18}  {iqr_t:>12}  {med_i:>16}  {iqr_i:>10}")

    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA C — Bubble chart BER vs flops_inf (tamaño = flops_train)
# ─────────────────────────────────────────────────────────────────────────────

def fig_bubble_flops(df: pd.DataFrame, dir_out: str):
    """
    Eje X : flops_inf mediana (escala log) — independiente del hardware
    Eje Y : BER media @ 10 dB
    Burbuja: área ∝ flops_train mediana

    Complementa/reemplaza fig9 (BER vs t_inf) para benchmarks reproducibles
    entre plataformas.
    """
    if not _tiene_flops(df):
        print("  [fig_bubble_flops] FLOPs no disponibles — figura omitida.")
        return

    eb_ref         = 10
    clfs_ordenados = ordenar_clasificadores_por_performance(df, eb_ref)
    sub            = df[(df["eb_n0"] == eb_ref) & df["confiable"]]
    ref_seed       = df[df["clf"] == "Bayes (ML)"].drop_duplicates(subset=["seed", "clf"])

    fig, ax = plt.subplots(figsize=(_W1 + 0.4, _H1 + 0.4))

    # Escala de burbujas: normalizar al máximo de flops_train presente
    ft_vals = [sub[sub["clf"] == c]["flops_train"].median()
               for c in clfs_ordenados if not sub[sub["clf"] == c].empty]
    ft_max = max((v for v in ft_vals if not np.isnan(v)), default=np.nan)
    _tiene_ft = not np.isnan(ft_max)

    texts = []
    for clf in clfs_ordenados:
        d = sub[sub["clf"] == clf]
        if d.empty:
            continue
        ber_med   = d["ber"].mean()
        fi_med    = d["flops_inf"].median()
        ft_med    = d["flops_train"].median()
        if np.isnan(fi_med):
            continue
        if _tiene_ft and not np.isnan(ft_med):
            bubble_sz = 30 + 500 * (ft_med / ft_max)
        else:
            bubble_sz = 180   # tamaño uniforme cuando flops_train no disponible
        sc = ax.scatter(fi_med, ber_med,
                        s=bubble_sz,
                        color=COLORES.get(clf, "gray"),
                        marker="o", alpha=0.75, zorder=3,
                        edgecolors="white", linewidths=0.5)
        txt = ax.text(fi_med, ber_med,
                      NOMBRES_CORTOS.get(clf, clf).replace("\n", " "),
                      fontsize=_FS_ANNOT, ha="left", va="bottom",
                      color=COLORES.get(clf, "gray"))
        texts.append(txt)

    if _HAS_ADJUSTTEXT:
        _adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.5))

    ax.set_xscale("log")
    ax.set_xlabel("FLOPs de inferencia por símbolo (mediana)")
    ax.set_ylabel("BER media @ 10 dB")
    if _tiene_ft:
        titulo_extra = "(área burbuja $\\propto$ FLOPs entrenamiento)"
    else:
        titulo_extra = "(tamaño uniforme — FLOPs entrenamiento no disponibles)"
    ax.set_title(f"Compromiso BER vs FLOPs\n{titulo_extra}")
    ax.yaxis.grid(True, which="both")
    ax.xaxis.grid(True, which="both")

    # Leyenda de escala de burbuja (solo cuando hay flops_train)
    if _tiene_ft:
        for frac, lbl in [(0.1, "10%"), (0.5, "50%"), (1.0, "100%")]:
            sz = 30 + 500 * frac
            ax.scatter([], [], s=sz, color="gray", alpha=0.5,
                       label=f"$f_{{train}}$: {lbl} del máx.")
        ax.legend(loc="upper right", fontsize=_FS_LEGEND - 1, framealpha=0.85)

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig11_bubble_ber_vs_flops", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA D — Eficiencia computacional η normalizada
# ─────────────────────────────────────────────────────────────────────────────

def imprimir_ranking_eficiencia(df: pd.DataFrame):
    """
    η = (1 - BER_clf/BER_Bayes) / log10(flops_inf)

    Cuanto mayor η, mejor BER por FLOP de inferencia.
    Métrica hardware-agnóstica para comparación entre plataformas.
    """
    if not _tiene_flops(df):
        print("\n[Eficiencia η] FLOPs no disponibles — ranking omitido.")
        print("               Disponible cuando flops_inf esté en los JSON.")
        return

    eb_ref = 10
    sub    = df[(df["eb_n0"] == eb_ref) & df["confiable"]]
    bayes  = sub[sub["clf"] == "Bayes (ML)"]["ber"].mean()
    if bayes <= 0 or np.isnan(bayes):
        print("\n[Eficiencia η] BER Bayes no disponible — ranking omitido.")
        return

    clfs_sin_bayes = [c for c in ORDEN_CLF
                      if c in df["clf"].unique() and c != "Bayes (ML)"]
    resultados = []
    for clf in clfs_sin_bayes:
        d       = sub[sub["clf"] == clf]
        fi_med  = d["flops_inf"].median()
        ber_med = d["ber"].mean()
        if np.isnan(fi_med) or fi_med <= 0 or np.isnan(ber_med):
            continue
        ber_norm = ber_med / bayes
        log_fi   = np.log10(fi_med)
        if log_fi <= 0:
            continue
        eta = (1.0 - ber_norm) / log_fi
        resultados.append((clf, ber_norm, fi_med, eta))

    if not resultados:
        print("\n[Eficiencia η] Sin datos suficientes para calcular η.")
        return

    resultados.sort(key=lambda x: -x[3])   # mayor η primero

    print("\n" + "=" * 75)
    print("  RANKING EFICIENCIA COMPUTACIONAL — η = (1 − BER_norm) / log₁₀(flops_inf)")
    print(f"  Referencia Bayes: BER = {bayes:.4e} @ {eb_ref} dB")
    print("=" * 75)
    print(f"  {'#':<3}  {'Clasificador':<20}  {'BER_norm':>10}  "
          f"{'flops_inf':>12}  {'η':>10}")
    print("  " + "-" * 71)
    for rank, (clf, ber_norm, fi, eta) in enumerate(resultados, 1):
        print(f"  {rank:<3}  {clf:<20}  {ber_norm:>10.4f}  {fi:>12.3e}  {eta:>10.4f}")
    print("=" * 75)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA D (figura) — Eficiencia η: bar chart hardware-agnóstico
# ─────────────────────────────────────────────────────────────────────────────

def fig_eficiencia_flops(df: pd.DataFrame, dir_out: str):
    """
    Bar chart de η = (1 − BER_norm) / log10(flops_inf) por clasificador.
    Omitido si flops_inf no está disponible en los JSON.
    """
    if not _tiene_flops(df):
        print("  [fig_eficiencia_flops] FLOPs no disponibles — figura omitida.")
        return

    eb_ref         = 10
    clfs_sin_bayes = [c for c in ORDEN_CLF
                      if c in df["clf"].unique() and c != "Bayes (ML)"]
    sub    = df[(df["eb_n0"] == eb_ref) & df["confiable"]]
    bayes  = sub[sub["clf"] == "Bayes (ML)"]["ber"].mean()
    if np.isnan(bayes) or bayes <= 0:
        print("  [fig_eficiencia_flops] BER Bayes no disponible — figura omitida.")
        return

    etas, colores, nombres = [], [], []
    for clf in clfs_sin_bayes:
        d      = sub[sub["clf"] == clf]
        fi_med = d["flops_inf"].median()
        ber    = d["ber"].mean()
        if np.isnan(fi_med) or fi_med <= 0 or np.isnan(ber):
            continue
        log_fi = np.log10(fi_med)
        if log_fi <= 0:
            continue
        etas.append((1.0 - ber / bayes) / log_fi)
        colores.append(COLORES.get(clf, "gray"))
        nombres.append(NOMBRES_CORTOS.get(clf, clf).replace("\n", " "))

    if not etas:
        return

    fig, ax = plt.subplots(figsize=(_W1 + 0.4, _H1))
    x = np.arange(len(etas))
    ax.bar(x, etas, color=colores, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=_FS_BASE - 1)
    ax.set_ylabel(r"$\eta = (1 - \mathrm{BER}_\mathrm{norm}) / \log_{10}(f_\mathrm{inf})$")
    ax.set_title(f"Eficiencia computacional $\\eta$ @ {eb_ref} dB\n"
                 "(mayor = mejor BER por FLOP, hardware-agnostico)")
    ax.yaxis.grid(True)
    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig_eficiencia_flops", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA E — Curvas de aprendizaje BER vs N_train
# ─────────────────────────────────────────────────────────────────────────────

def fig_curvas_aprendizaje(df: pd.DataFrame, dir_out: str):
    """
    BER vs N_train por clasificador — requiere que el JSON incluya resultados
    con N_train variable (campo 'n_train' por corrida o por registro).

    TODO: implementar cuando el pipeline principal (curvas_aprendizaje.py)
          agregue los resultados al JSON de benchmark. Por ahora:
          - Si existe columna 'n_train' con > 1 valor distinto → graficar.
          - Si no → omitir con aviso.

    Uso previsto (Trabajo Futuro): determinar tamaño mínimo de secuencia piloto.
    """
    if "n_train" not in df.columns or df["n_train"].nunique() <= 1:
        print("  [fig_curvas_aprendizaje] Campo 'n_train' variable no disponible"
              " en los JSON — figura omitida.")
        print("  TODO: integrar curvas_aprendizaje.py al JSON de benchmark"
              " para habilitar esta figura.")
        return

    eb_ref         = 10
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()
                      and c != "Bayes (ML)"]
    ber_teo        = _ber_teorica_16qam(np.array([float(eb_ref)]))[0]
    ns_disponibles = sorted(df["n_train"].dropna().unique())

    fig, ax = plt.subplots(figsize=(_W1 + 0.4, _H1 + 0.2))

    for clf in clfs_presentes:
        medianas = []
        for n in ns_disponibles:
            sub = df[(df["clf"] == clf) & (df["eb_n0"] == eb_ref)
                     & (df["n_train"] == n) & df["confiable"]]
            if sub.empty:
                medianas.append(np.nan)
            else:
                medianas.append(np.median(sub["ber"].values))
        ax.plot(ns_disponibles, medianas,
                color=COLORES.get(clf, "gray"),
                marker=MARKERS.get(clf, "o"),
                ms=4, lw=1.3, label=clf)

    ax.axhline(ber_teo, color="black", ls="--", lw=1.0,
               label=f"Bayes teórico ({ber_teo:.2e})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N_{train}$ (símbolos)")
    ax.set_ylabel(f"BER mediana @ {eb_ref} dB")
    ax.set_title("Curvas de aprendizaje — BER vs. $N_{{train}}$\n"
                 "(tamaño mínimo de secuencia piloto)")
    ax.legend(loc="upper right", ncol=2, fontsize=_FS_LEGEND)
    ax.yaxis.grid(True, which="both")
    ax.xaxis.grid(True, which="both")

    fig.tight_layout(pad=0.4)
    _guardar(fig, "fig12_curvas_aprendizaje", dir_out)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA F — Corrección por comparaciones múltiples (Benjamini-Hochberg)
# ─────────────────────────────────────────────────────────────────────────────

def _bh_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Corrección Benjamini-Hochberg (FDR) sobre un array 1-D de p-values.
    Devuelve array de p_adj del mismo tamaño, con NaN donde p_values era NaN.
    """
    mask_valid = ~np.isnan(p_values)
    p_adj = np.full_like(p_values, np.nan)
    if mask_valid.sum() == 0:
        return p_adj
    m       = mask_valid.sum()
    idx_ord = np.argsort(p_values[mask_valid])
    p_sorted = p_values[mask_valid][idx_ord]
    ranks   = np.arange(1, m + 1)
    # p_adj[i] = min over j>=i of (m/rank[j]) * p[j], clipped to [0,1]
    adj = np.minimum(1.0, np.minimum.accumulate((m / ranks * p_sorted)[::-1])[::-1])
    p_adj_valid              = np.empty(m)
    p_adj_valid[idx_ord]     = adj
    p_adj[mask_valid]        = p_adj_valid
    return p_adj


def imprimir_tabla_pvalues_adj(df: pd.DataFrame, eb_ref: int = 10):
    """
    Imprime tabla de p-values (t de Student pareado sobre Delta_dB) y p_adj
    (Benjamini-Hochberg) de cada clasificador vs Bayes.

    Se usa ttest_1samp sobre diferencias por seed:
        Delta_dB = 10*log10(BER_clf / BER_Bayes)
    Coherente con el IC 95% declarado en el articulo. Las diferencias en
    escala logaritmica son mas simetricas que las BER crudas, mejorando
    la validez del supuesto de normalidad del t-test.
    Con >=9 comparaciones multiples, BH controla el FDR al 5%.
    """
    n_corridas     = df["seed"].nunique()
    clfs_sin_bayes = [c for c in ORDEN_CLF
                      if c in df["clf"].unique() and c != "Bayes (ML)"]

    tests   = [test_pareado(df, clf, "Bayes (ML)", eb=eb_ref) for clf in clfs_sin_bayes]
    p_t_arr = np.array([t["p_ttest"]    for t in tests])
    p_w_arr = np.array([t["p_wilcoxon"] for t in tests])
    p_adj_t = _bh_correction(p_t_arr)
    p_adj_w = _bh_correction(p_w_arr)

    print(f"\n{'='*90}")
    print(f"  P-VALUES vs Bayes — t Student pareado & Wilcoxon (Delta_dB) + Benjamini-Hochberg")
    print(f"  Eb/N0 = {eb_ref} dB  |  n corridas = {n_corridas}  |  "
          f"alpha = 0.05  |  m = {len(clfs_sin_bayes)} comparaciones")
    print(f"{'='*90}")
    print(f"  {'Clasificador':<20}  {'Delta_dB':>9}  {'+-IC95%':>7}  "
          f"{'p_t':>8}  {'p_t_BH':>8}  {'p_W':>8}  {'p_W_BH':>8}  {'Sig.':>6}")
    print("  " + "-" * 86)
    for clf, t, pt, ptadj, pw, pwadj in zip(
            clfs_sin_bayes, tests, p_t_arr, p_adj_t, p_w_arr, p_adj_w):
        dm   = f"{t['delta_media']:+.3f}" if not np.isnan(t['delta_media'])  else "N/A"
        dic  = f"{t['delta_ic95']:.3f}"  if not np.isnan(t['delta_ic95'])   else "N/A"
        spt  = f"{pt:.4f}"   if not np.isnan(pt)    else "N/A"
        spta = f"{ptadj:.4f}" if not np.isnan(ptadj) else "N/A"
        spw  = f"{pw:.4f}"   if not np.isnan(pw)    else "N/A"
        spwa = f"{pwadj:.4f}" if not np.isnan(pwadj) else "N/A"
        sig  = "(*)" if (not np.isnan(ptadj) and ptadj < 0.05) else ""
        print(f"  {clf:<20}  {dm:>9}  {dic:>7}  "
              f"{spt:>8}  {spta:>8}  {spw:>8}  {spwa:>8}  {sig:>6}")
    print(f"{'='*90}")
    print("  Sig. (*): p_t_BH < 0.05 — diferencia significativa vs Bayes (FDR 5%).")
    print("  Delta_dB = 10*log10(BER_clf/BER_Bayes), positivo = peor que Bayes.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ANÁLISIS ESTADÍSTICO — BENCHMARK ML 16-QAM  [IEEE-ready]")
    print("=" * 60)

    os.makedirs(DIR_SALIDA, exist_ok=True)
    df, eb_n0_validos = cargar_datos()

    global EB_N0_ESPERADOS
    EB_N0_ESPERADOS = eb_n0_validos

    imprimir_tabla_resumen(df)

    # Tarea B — Tabla FLOPs
    imprimir_tabla_flops(df)

    # Tarea D — Ranking eficiencia η
    imprimir_ranking_eficiencia(df)

    # Tarea F — P-values corregidos Benjamini-Hochberg
    imprimir_tabla_pvalues_adj(df, eb_ref=10)

    print(f"\n[Figuras] Generando en: {DIR_SALIDA}\n")
    fig_ber_vs_ebn0(df, DIR_SALIDA)
    fig_boxplot_ber(df, DIR_SALIDA)
    fig_degradacion_db(df, DIR_SALIDA)
    fig_heatmap_pvalues(df, DIR_SALIDA)
    fig_boxplot_ttrain(df, DIR_SALIDA)
    fig_ttrain_vs_ebn0(df, DIR_SALIDA)
    fig_boxplot_tinf(df, DIR_SALIDA)
    fig_scatter_ber_ttrain(df, DIR_SALIDA)
    fig_scatter_ber_tinf(df, DIR_SALIDA)
    fig_cv(df, DIR_SALIDA)

    # Tarea C — Bubble chart FLOPs (omitido si flops no disponibles)
    fig_bubble_flops(df, DIR_SALIDA)

    # Tarea D (figura) — Eficiencia η (omitido si flops no disponibles)
    fig_eficiencia_flops(df, DIR_SALIDA)

    # Tarea E — Curvas de aprendizaje (omitido si n_train variable no disponible)
    fig_curvas_aprendizaje(df, DIR_SALIDA)

    n_figs = 10
    if _tiene_flops(df):
        n_figs += 2   # bubble + eficiencia
    if "n_train" in df.columns and df["n_train"].nunique() > 1:
        n_figs += 1

    formatos = []
    if GUARDAR_PNG: formatos.append("PNG 600 dpi")
    if GUARDAR_PDF: formatos.append("PDF vectorial")
    print(f"\n[Listo] {n_figs} figuras — {df['seed'].nunique()} corridas — '{DIR_SALIDA}'")
    print(f"        Formatos: {', '.join(formatos)}")
    print("=" * 60)


if __name__ == "__main__":
    main()