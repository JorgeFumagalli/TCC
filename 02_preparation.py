# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:37:47 2025

@author: jorge
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configurações
# =============================================================================

DATA_DIR = Path(r"C:\Users\jorge\Downloads")
DATA_FILE = "macro_credito_bcb_com_credito_pib_reconstruido.csv"
OUTPUT_DIR = Path(r"C:\Users\jorge\Downloads\model_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_RATE_COLUMNS = [
    "inadimpl_cartao_total",    # Inadimplência cartão total
    "inadimpl_cartao_rot",      # Inadimplência cartão rotativo
    "inadimpl_cartao_parc",     # Inadimplência cartão parcelado
    "inadimplencia_familias",   # Inadimplência das famílias
]

# =============================================================================
# Funções de preparação de dados
# =============================================================================

def load_and_prepare_data(file_path, target_col):
    """Carrega e prepara os dados para modelagem"""
    df = pd.read_csv(file_path)
    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values('data').reset_index(drop=True)
    
    # Remove colunas com muitos NaN
    df_clean = df.dropna(thresh=len(df)*0.7, axis=1)
    
    # Remove linhas com NaN no target
    df_clean = df_clean.dropna(subset=[target_col])
    
    # Forward fill para preencher NaN restantes
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill')
    
    return df_clean

def create_features(df, target_col, lookback=3):
    """Cria features de lag e rolling statistics"""
    feature_df = df.copy()
    
    # Features de lag
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'data':
            for lag in range(1, lookback + 1):
                feature_df[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)
    
    # Rolling statistics
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'data':
            feature_df[f'{col}_rolling_mean_3'] = feature_df[col].rolling(3).mean()
            feature_df[f'{col}_rolling_std_3'] = feature_df[col].rolling(3).std()
    
    # Remove linhas com NaN criados pelos lags
    feature_df = feature_df.dropna()
    
    return feature_df

def prepare_sequences(data, target_col, sequence_length=12):
    """Prepara sequências para modelos LSTM"""
    feature_cols = [col for col in data.columns if col not in ['data', target_col]]
    
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[feature_cols].iloc[i-sequence_length:i].values)
        y.append(data[target_col].iloc[i])
    
    return np.array(X), np.array(y)

def create_classification_target(y, method='direction', threshold=None):
    """
    Converte problema de regressão em classificação
    
    Methods:
    - 'direction': 1 se valor aumenta, 0 se diminui
    - 'threshold': 1 se valor > threshold, 0 caso contrário
    - 'quantile': 1 se valor > percentil 50, 0 caso contrário
    """
    if method == 'direction':
        # Classifica se a série vai subir (1) ou descer (0)
        y_diff = np.diff(y)
        y_class = (y_diff > 0).astype(int)
        return y_class
    
    elif method == 'threshold' and threshold is not None:
        return (y > threshold).astype(int)
    
    elif method == 'quantile':
        threshold = np.percentile(y, 50)
        return (y > threshold).astype(int)
    
    else:
        raise ValueError("Método não reconhecido ou threshold não fornecido")

def analyze_target_variable(df, target_col):
    """Analisa uma variável target específica"""
    print(f"\n=== ANÁLISE DA VARIÁVEL: {target_col} ===")
    
    if target_col not in df.columns:
        print(f"Variável {target_col} não encontrada nos dados!")
        return None
    
    # Estatísticas básicas
    target_data = df[target_col].dropna()
    print(f"Observações disponíveis: {len(target_data)}")
    print(f"Período: {df['data'].min()} a {df['data'].max()}")
    print(f"Média: {target_data.mean():.4f}")
    print(f"Desvio padrão: {target_data.std():.4f}")
    print(f"Min: {target_data.min():.4f}, Max: {target_data.max():.4f}")
    
    return target_data

def prepare_data_for_modeling(target_variable):
    """Função principal para preparar dados para uma variável específica"""
    print(f"\n{'='*80}")
    print(f"PREPARANDO DADOS PARA: {target_variable}")
    print(f"{'='*80}")
    
    # Carregar dados
    data_path = DATA_DIR / DATA_FILE
    if not data_path.exists():
        print(f"Arquivo {data_path} não encontrado.")
        return None
    
    # Preparar dados para esta variável específica
    df = load_and_prepare_data(data_path, target_variable)
    
    # Verificar se a variável existe e tem dados suficientes
    target_analysis = analyze_target_variable(df, target_variable)
    if target_analysis is None or len(target_analysis) < 50:
        print(f"Dados insuficientes para {target_variable}. Pulando...")
        return None
    
    # Preparar features
    print(f"\nPreparando features para {target_variable}...")
    feature_df = create_features(df, target_variable, lookback=3)
    print(f"Features criadas: {len(feature_df)} observações, {len(feature_df.columns)} colunas")
    
    # Dividir dados
    feature_cols = [col for col in feature_df.columns if col not in ['data', target_variable]]
    X = feature_df[feature_cols].values
    y = feature_df[target_variable].values
    dates = feature_df['data'].values
    
    # Split temporal (80% treino, 20% teste)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Treino: {len(X_train)} observações")
    print(f"Teste: {len(X_test)} observações")
    
    # Preparar targets de classificação
    print(f"\nPreparando targets de classificação para {target_variable}...")
    y_train_class = create_classification_target(y_train, method='direction')
    y_test_class = create_classification_target(y_test, method='direction')
    
    # Ajustar tamanhos (direction remove 1 elemento)
    X_train_class = X_train_scaled[1:]
    X_test_class = X_test_scaled[1:]
    dates_test_class = dates_test[1:]
    
    print(f"Classificação - Treino: {len(X_train_class)} observações")
    print(f"Classificação - Teste: {len(X_test_class)} observações")
    print(f"Distribuição classes teste: {np.bincount(y_test_class)}")
    
    # Preparar dados para LSTM
    sequence_length = 12
    X_seq, y_seq = prepare_sequences(feature_df, target_variable, sequence_length)
    
    split_seq = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
    y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]
    
    # Normalizar sequências
    scaler_seq = MinMaxScaler()
    X_train_seq_scaled = scaler_seq.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
    X_test_seq_scaled = scaler_seq.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
    
    dates_lstm = dates[split_seq + sequence_length:]
    
    return {
        'target_variable': target_variable,
        'feature_df': feature_df,
        'feature_cols': feature_cols,
        
        # Dados de regressão
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'dates_test': dates_test,
        
        # Dados de classificação
        'X_train_class': X_train_class,
        'X_test_class': X_test_class,
        'y_train_class': y_train_class,
        'y_test_class': y_test_class,
        'dates_test_class': dates_test_class,
        
        # Dados para LSTM
        'X_train_seq': X_train_seq_scaled,
        'X_test_seq': X_test_seq_scaled,
        'y_train_seq': y_train_seq,
        'y_test_seq': y_test_seq,
        'dates_lstm': dates_lstm,
        'sequence_length': sequence_length,
        
        # Scalers
        'scaler': scaler,
        'scaler_seq': scaler_seq
    }

if __name__ == "__main__":
    # Teste da preparação de dados
    for target_var in DEFAULT_RATE_COLUMNS:
        data_dict = prepare_data_for_modeling(target_var)
        if data_dict:
            print(f"Dados preparados com sucesso para {target_var}")
        else:
            print(f"Falha na preparação dos dados para {target_var}")
