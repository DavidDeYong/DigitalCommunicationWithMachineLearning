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
from scipy.stats import wilcoxon

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
    "SVM RBF":       "#C0392B",   # rojo oscuro
    "SVM Lineal":    "#E67E22",   # naranja
    "MLP Simple":    "#7D3C98",   # púrpura
    "Red Profunda":  "#1A8599",   # cian oscuro
    "SGD-Nystroem":  "#5D4037",   # marrón
    "ELM":           "#1E8449",   # verde oscuro
    "Logistic Reg.": "#B7950B",   # dorado oscuro
}

# Marcadores variados para distinguir curvas sin depender solo del color
MARKERS = {
    "Bayes (ML)":    "D",    # diamante
    "SVM RBF":       "s",    # cuadrado
    "SVM Lineal":    "^",    # triángulo arriba
    "MLP Simple":    "v",    # triángulo abajo
    "Red Profunda":  "o",    # círculo
    "SGD-Nystroem":  "P",    # plus grueso
    "ELM":           "X",    # X gruesa
    "Logistic Reg.": "h",    # hexágono
}

ORDEN_CLF = list(COLORES.keys())

NOMBRES_CORTOS = {
    "Bayes (ML)":    "Bayes",
    "SVM RBF":       "SVM\nRBF",
    "SVM Lineal":    "SVM\nLineal",
    "KNN":           "KNN",
    "Random Forest": "Rand.\nForest",
    "MLP Simple":    "MLP\nSimple",
    "Red Profunda":  "Red\nProf.",
    "SGD-Nystroem":  "SGD-\nNystr.",
    "ELM":           "ELM",
    "Logistic Reg.": "Log.\nReg.",
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
CLFS_EXCLUIR   = {"KNN", "KNN (k=15)", "Random Forest"}
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
            n = len(eb_n0_list)
            for i in range(n):
                registros.append({
                    "seed":       seed,
                    "corrida_id": corrida_id,
                    "fecha":      fecha,
                    "clf":        clf,
                    "eb_n0":      int(eb_n0_list[i]),
                    "ber":        float(ber_list[i])       if i < len(ber_list)       else np.nan,
                    "accuracy":   float(acc_list[i])       if i < len(acc_list)       else np.nan,
                    "t_train":    float(t_train_list[i])   if i < len(t_train_list)   else np.nan,
                    "t_inf":      float(t_inf_list[i])     if i < len(t_inf_list)     else np.nan,
                    "confiable":  bool(confiable_list[i])  if i < len(confiable_list) else False,
                    "n_errores":  int(errores_list[i])     if i < len(errores_list)   else 0,
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


def cargar_datos() -> tuple[pd.DataFrame, list]:
    print("\n[Carga] Buscando fuente de datos...")
    if os.path.exists(RUTA_JSON):
        print(f"  Fuente: JSON → {RUTA_JSON}")
        df = cargar_desde_json(RUTA_JSON, seeds_excluir=SEEDS_EXCLUIR)
        fuente = "JSON"
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
# FIGURA 4 — Heatmap p-values Wilcoxon
# ─────────────────────────────────────────────────────────────────────────────

def fig_heatmap_pvalues(df: pd.DataFrame, dir_out: str):
    eb_ref         = 10
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    n              = len(clfs_presentes)
    sub            = df[(df["eb_n0"] == eb_ref) & df["confiable"]]
    p_matrix       = np.ones((n, n))

    for i, clf_a in enumerate(clfs_presentes):
        for j, clf_b in enumerate(clfs_presentes):
            if i == j:
                continue
            vals_a  = sub[sub["clf"] == clf_a]["ber"].values
            vals_b  = sub[sub["clf"] == clf_b]["ber"].values
            min_n   = min(len(vals_a), len(vals_b))
            if min_n < 2:
                p_matrix[i, j] = np.nan
                continue
            try:
                _, p = wilcoxon(vals_a[:min_n], vals_b[:min_n])
            except Exception:
                p = 1.0
            p_matrix[i, j] = p

    mask = np.eye(n, dtype=bool)
    p_df = pd.DataFrame(p_matrix, index=clfs_presentes, columns=clfs_presentes)

    # Tamaño: cuadrado compacto para publicación
    fig, ax = plt.subplots(figsize=(_W1 + 0.5, _W1 + 0.5))
    cmap    = sns.diverging_palette(10, 145, s=80, l=55, as_cmap=True)
    sns.heatmap(p_df, annot=True, fmt=".3f", cmap=cmap,
                vmin=0, vmax=0.1, mask=mask,
                linewidths=0.4, ax=ax,
                annot_kws={"size": _FS_ANNOT - 1},
                cbar_kws={"label": "$p$-value (Wilcoxon)", "shrink": 0.8})

    ax.set_title(f"$p$-values Wilcoxon — BER @ {eb_ref} dB",
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
        ber_media   = d["ber"].mean()
        train_media = d["t_train"].median()
        ber_std     = d["ber"].std()
        train_iqr   = d["t_train"].quantile(0.75) - d["t_train"].quantile(0.25)

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
        ber_media = d["ber"].mean()
        inf_media = d["t_inf"].median()
        ber_std   = d["ber"].std()
        inf_iqr   = d["t_inf"].quantile(0.75) - d["t_inf"].quantile(0.25)

        ax.errorbar(inf_media, ber_media,
                    xerr=inf_iqr / 2, yerr=ber_std,
                    fmt=MARKERS.get(clf, "o"),
                    color=COLORES.get(clf, "gray"),
                    ms=5, capsize=2.5, lw=1.0,
                    label=clf, zorder=3)

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
    clfs_presentes = [c for c in ORDEN_CLF if c in df["clf"].unique()]
    sub = df[df["eb_n0"].isin(EB_N0_ESTABLES) & df["confiable"]]
    print("\n" + "=" * 90)
    print("  TABLA RESUMEN — Media ± IC95% — Eb/N0 estables")
    print("=" * 90)
    print(f"  {'Clasificador':<20}  {'BER media':>12}  {'±IC95%':>10}  "
          f"{'CV_BER%':>8}  {'t_train(s)':>11}  {'t_inf(µs)':>11}  {'n':>4}")
    print("  " + "-" * 86)
    for clf in clfs_presentes:
        d  = sub[sub["clf"] == clf]
        eb = estadisticas_grupo(d, "ber")
        et = estadisticas_grupo(d, "t_train")
        ei = estadisticas_grupo(d, "t_inf")
        if not eb:
            continue
        ic_ancho = (eb["ic_hi"] - eb["ic_lo"]) / 2
        print(f"  {clf:<20}  {eb['media']:>12.4e}  {ic_ancho:>10.2e}  "
              f"{eb['cv']:>8.1f}  {et.get('mediana', 0):>11.2f}  "
              f"{ei.get('mediana', 0):>11.1f}  {eb['n']:>4}")
    print("=" * 90)


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

    formatos = []
    if GUARDAR_PNG: formatos.append("PNG 600 dpi")
    if GUARDAR_PDF: formatos.append("PDF vectorial")
    print(f"\n[Listo] 10 figuras guardadas en '{DIR_SALIDA}'")
    print(f"        Formatos: {', '.join(formatos)}")
    print("=" * 60)


if __name__ == "__main__":
    main()