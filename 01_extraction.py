# -*- coding: utf-8 -*-
"""
TCC – Risco de Crédito em Cartões × Macroeconomia (BCB/SGS)
Versão completa e estável para Spyder/Anaconda

- Baixa séries do SGS/BCB com validação de JSON e fallback de host
- Normaliza datas para o 1º dia de cada mês
- Reconstrói Crédito/PIB (%) usando:
    * 20786 (Crédito total do SFN, R$ mi, mensal)
    * 4380  (PIB nominal IBGE, R$ mi, trimestral -> mensal via forward-fill)
- Salva em C:/Users/jorge/Downloads (CSV/XLSX)
- Plots opcionais
"""
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from functools import reduce

# =============================================================================
# Configurações
# =============================================================================

OUTPUT_DIR = Path(r"C:\Users\jorge\Downloads")
OUTPUT_DIR.mkdir(exist_ok=True)

# Séries mensais diretas
SERIES = {
    "selic_mensal": 4390,        # SELIC acumulada no mês (% a.m.)
    "ibcbr_dessaz": 24364,       # IBC-Br dessazonalizado (índice)
    "ibcbr_sem_ajuste": 24363,   # IBC-Br sem ajuste sazonal (índice)
    # Inadimplência de cartão (% do saldo da carteira)
    "inadimpl_cartao_total": 25464,
    "inadimpl_cartao_rot":   25465,
    "inadimpl_cartao_parc":  25466,
    "ipca_mensal": 433,          # IPCA - Índice Nacional de Preços ao Consumidor Amplo (% a.m.)
    "pib_real_trimestral": 7326, # PIB - valores encadeados do trimestre anterior (var. % trimestral)
    "comprometimento_renda": 19882, # Comprometimento de renda das famílias com o SFN (%) - até 2021
    "endividamento_familias": 19881, # Endividamento das famílias com o SFN (%) - pode ter dados mais recentes
    "inadimplencia_familias": 21082, # Taxa de inadimplência das famílias (%) - alternativa mais recente
}

# =============================================================================
# Sessão HTTP + utilitários
# =============================================================================

SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (SGS-client; TCC Jorge Fumagalli)"
})

def _fmt_bcb_date(d):
    if d is None or d == "":
        return None
    return pd.to_datetime(d).strftime("%d/%m/%Y")

def _fetch_series(host_base, codigo, start, end, timeout=60):
    url = f"{host_base}/dados/serie/bcdata.sgs.{codigo}/dados"
    params = {"formato": "json"}
    if start: params["dataInicial"] = _fmt_bcb_date(start)
    if end:   params["dataFinal"]   = _fmt_bcb_date(end)
    r = SESSION.get(url, params=params, timeout=timeout)
    print(f"[SGS] {r.status_code} | {r.headers.get('Content-Type','')} | {r.url}")
    return r

def get_sgs(codigo, start="2015-01-01", end=None, timeout=60):
    """
    Baixa série SGS em DataFrame [data, valor] com:
    - validação de JSON,
    - fallback de host: api.bcb.gov.br -> dadosabertos.bcb.gov.br.
    """
    hosts = ["https://api.bcb.gov.br", "https://dadosabertos.bcb.gov.br"]
    last_diag = None
    for host in hosts:
        r = _fetch_series(host, codigo, start, end, timeout=timeout)
        ctype = (r.headers.get("Content-Type") or "").lower()

        if r.status_code != 200 or "json" not in ctype:
            snippet = (r.text or "")[:300].replace("\n", " ")
            last_diag = f"Status={r.status_code} | CT={ctype} | Body~ {snippet}"
            continue

        try:
            data = r.json()
        except Exception:
            snippet = (r.text or "")[:300].replace("\n", " ")
            last_diag = f"JSON decode falhou | Status={r.status_code} | CT={ctype} | Body~ {snippet}"
            continue

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=["data", "valor"])
        df["valor"] = pd.to_numeric(df["valor"].astype(str).str.replace(",", "."), errors="coerce")
        df["data"]  = pd.to_datetime(df["data"], dayfirst=True)
        return df

    raise RuntimeError(
        f"Falha ao obter JSON para a série {codigo}. Último diagnóstico: {last_diag}"
    )

def to_month_start(df):
    """Converte a data para o primeiro dia do mês (compatível com pandas recente)."""
    df = df.copy()
    df["data"] = df["data"].dt.to_period("M").dt.start_time
    return df

# ---- funções para reconstrução de Crédito/PIB ----

def expand_quarter_to_months(df_q, how="ffill"):
    """
    Expande uma série trimestral ['data','valor'] para mensal (MS) usando forward-fill
    dentro do trimestre. Retorna ['data','valor'] mensal (início de mês).
    """
    s = df_q.set_index("data")["valor"].sort_index()
    s_m = s.resample("MS").ffill() if how == "ffill" else s.resample("MS").bfill()
    out = s_m.reset_index()
    return out[["data", "valor"]]

def compute_credit_to_gdp(credito_df_m, pib_df_m, col_cred='valor', col_pib='valor'):
    """
    crédito/PIB (%) = (credito / pib) * 100
    - credito_df_m: ['data', col_cred] em R$ mi (mensal)
    - pib_df_m:     ['data', col_pib]   em R$ mi (mensal)
    Retorna: ['data','credito_total_rs','pib_nominal_rs','credito_pct_pib_calc']
    """
    # Renomeia ANTES do merge para evitar valor_x/valor_y
    c = credito_df_m[['data', col_cred]].rename(columns={col_cred: 'credito_total_rs'})
    p = pib_df_m[['data', col_pib]].rename(columns={col_pib: 'pib_nominal_rs'})

    df = pd.merge(c, p, on='data', how='outer').sort_values('data')

    # Cálculo
    df['credito_pct_pib_calc'] = (df['credito_total_rs'] / df['pib_nominal_rs']) * 100
    return df[['data', 'credito_total_rs', 'pib_nominal_rs', 'credito_pct_pib_calc']]

# =============================================================================
# Execução principal
# =============================================================================

if __name__ == "__main__":
    # 1) Séries mensais diretas
    dfs = []
    for colname, code in SERIES.items():
        print(f"\nBaixando {colname} (código {code})...")
        try:
            df = get_sgs(code, start=START, end=END)
        except Exception as e:
            print(f"[ERRO] Série {colname} ({code}) falhou: {e}")
            continue

        if df.empty:
            print(f"[AVISO] Série {colname} ({code}) veio vazia.")
            continue

        df = to_month_start(df).rename(columns={"valor": colname})
        dfs.append(df[["data", colname]])

    # 2) Reconstruir Crédito/PIB (%)
    print("\nBaixando séries para reconstruir Crédito/PIB...")
    credito_df = get_sgs(20786, start=START, end=END)   # mensal, R$ mi
    pib_q_df   = get_sgs(4380,   start=START, end=END)   # trimestral, R$ mi

    credito_m = to_month_start(credito_df)          # ['data','valor']
    pib_m     = expand_quarter_to_months(pib_q_df)  # ['data','valor'] mensal por ffill

    credito_pib = compute_credit_to_gdp(credito_m, pib_m)  # colunas de cálculo incluídas

    # 3) Consolidar tudo
    all_frames = dfs + [credito_pib]
    if not all_frames:
        raise ValueError("Nenhuma série foi baixada com sucesso. Verifique códigos/período.")

    base = reduce(lambda l, r: pd.merge(l, r, on="data", how="outer"), all_frames)
    base = base.sort_values("data").reset_index(drop=True)

    # 4) Salvar
    out_csv  = OUTPUT_DIR / "macro_credito_bcb_com_credito_pib_reconstruido.csv"
    out_xlsx = OUTPUT_DIR / "macro_credito_bcb_com_credito_pib_reconstruido.xlsx"
    base.to_csv(out_csv, index=False, encoding="utf-8")
    base.to_excel(out_xlsx, index=False)

    print("\n--- RESULTADO ---")
    print("CSV :", out_csv)
    print("XLSX:", out_xlsx)
    print("\nPrévia:")
    print(base.head())

    # 5) Plot opcional
    if True:
        cols_plot = [
            "selic_mensal",
            "ibcbr_dessaz",
            "ibcbr_sem_ajuste",
            "credito_total_rs",
            "pib_nominal_rs",
            "credito_pct_pib_calc",
            "inadimpl_cartao_total",
            "inadimpl_cartao_rot",
            "inadimpl_cartao_parc",
            "ipca_mensal",
            "pib_real_trimestral",
            "comprometimento_renda",
            "endividamento_familias",
            "inadimplencia_familias",
        ]
        for c in [c for c in cols_plot if c in base.columns]:
            plt.figure()
            plt.plot(base["data"], base[c])
            plt.title(c)
            plt.xlabel("Data")
            plt.ylabel(c)
            plt.tight_layout()
        plt.show()
