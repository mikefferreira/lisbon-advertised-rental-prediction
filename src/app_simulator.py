import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import folium
from streamlit_folium import st_folium
import json
import unicodedata
import branca.colormap as cm # Biblioteca para escalas de cores avançadas
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURAÇÃO VISUAL PREMIUM
# ==========================================
st.set_page_config(page_title="Urban Pressure Audit | Policy Simulator", layout="wide")
st.markdown("""
    <style>
    /* 1. PUXAR TUDO PARA CIMA */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
    }
    
    /* Esconder elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #F8F9FA;
    }

    /* Título principal colado ao topo */
    .main-title {
        margin-top: -4rem;
        font-weight: 800;
        color: #1E293B;
    }

    /* Estilo dos Cartões de Métricas */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .metric-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 8px;
        display: block;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
# ==========================================
# 2. CARREGAR MOTOR
# ==========================================
@st.cache_resource
def load_system():
    # Usa o MODELS_DIR que definimos acima
    modelo = joblib.load(MODELS_DIR / 'augmented_lightgbm_model.pkl')
    dados = pd.read_pickle(MODELS_DIR / 'X_test_aug.pkl')
    dicionario = joblib.load(MODELS_DIR / 'dicionario_freguesias.pkl')
    return modelo, dados, dicionario

lgbm_final, X_test_aug, freguesias_dict = load_system()

# 3. MENU LATERAL COM NOMES REAIS
st.sidebar.caption("📅 Cenário Base: Novembro de 2025 (Dados Reais do Mercado)") # <--- ADICIONA ISTO

# Lê o teu dicionário dinâmico
lista_freguesias = sorted(list(freguesias_dict.keys()))
freguesia_selecionada = st.sidebar.selectbox("📍 Selecione a Freguesia Alvo", lista_freguesias)
idx_alvo = freguesias_dict[freguesia_selecionada]
observacao_base = X_test_aug.loc[[idx_alvo]].copy()

# (Logo a seguir a calcular idx_alvo e observacao_base...)

st.sidebar.markdown("### 📊 Opções Mapa")
modo_mapa = st.sidebar.radio(
    "Tipo de Dados a Visualizar:",
    ('Preço Absoluto Simulado (€/m²)', 'Impacto da Política (Delta %)')
)


st.sidebar.markdown("### 🎛️ Alavancas Políticas")

val_al = float(observacao_base.get('Densidade_AL_km2', 0.0))
val_reab_taxa = float(observacao_base.get('Taxa_Conversao_Reabilitacao', 0.0))
val_alv_reab = float(observacao_base.get('Stock_12M_ALV_Reabilitacao_Count', 0.0))

# 2. O SEGREDO MÁXIMO: Atualizar a memória se mudarmos de Freguesia
if 'ultima_freguesia' not in st.session_state or st.session_state['ultima_freguesia'] != freguesia_selecionada:
    st.session_state['ultima_freguesia'] = freguesia_selecionada
    st.session_state['slider_al'] = val_al
    st.session_state['slider_vol'] = val_alv_reab
    st.session_state['slider_taxa'] = val_reab_taxa

# 2. CRIAR A FUNÇÃO DE CALLBACK (Obrigatório para o botão funcionar bem)
def reset_alavancas():
    st.session_state['slider_al'] = val_al
    st.session_state['slider_vol'] = val_alv_reab
    st.session_state['slider_taxa'] = val_reab_taxa

st.sidebar.button("🔄 Repor Valores Base (Reset)", on_click=reset_alavancas, use_container_width=True)

with st.sidebar.expander("Dinâmicas de Turismo", expanded=True):
    nova_densidade_al = st.slider(
        "Densidade de AL (unid/km²)", 
        0.0, max(1500.0, val_al*2 if val_al > 0 else 100.0), val_al, step=10.0,
        key='slider_al'
    )

with st.sidebar.expander("Reabilitação Urbana", expanded=True):
    novo_alv_reab = st.slider(
        label="Alvarás de Reabilitação Emitidos", 
        min_value=0.0, 
        max_value=max(150.0, val_alv_reab*3), 
        value=val_alv_reab, 
        step=1.0,
        key='slider_vol',
        help="Representa o 'Stock a 12 Meses': o volume total de licenças (Alvarás) aprovadas pela Câmara no último ano. Aumentar este valor simula um cenário onde a burocracia é mais rápida, injetando mais habitação renovada no mercado da freguesia."
    )

    nova_taxa_reab = st.slider(
        "Taxa de Conversão (Eficiência)", 
        0.0, 1.0, val_reab_taxa, step=0.05,
        key='slider_taxa',
        help="Mede a velocidade e capacidade da Câmara em transformar Pedidos em Alvarás reais. 1.0 significa 100% de eficiência (zero bloqueios burocráticos)."
    )



# 4. INFERÊNCIA MATEMÁTICA
pred_log_base = lgbm_final.predict(observacao_base)
preco_base = np.expm1(pred_log_base)[0]

observacao_simulada = observacao_base.copy()
if 'Densidade_AL_km2' in observacao_simulada: observacao_simulada['Densidade_AL_km2'] = nova_densidade_al
if 'Taxa_Conversao_Reabilitacao' in observacao_simulada: observacao_simulada['Taxa_Conversao_Reabilitacao'] = nova_taxa_reab
if 'Stock_12M_ALV_Reabilitacao_Count' in observacao_simulada: observacao_simulada['Stock_12M_ALV_Reabilitacao_Count'] = novo_alv_reab

pred_log_sim = lgbm_final.predict(observacao_simulada)
preco_simulado = np.expm1(pred_log_sim)[0]

impacto_abs = preco_simulado - preco_base
impacto_pct = (impacto_abs / preco_base) * 100 if preco_base > 0 else 0

# ==========================================
# 5. ECRÃ CENTRAL
# ==========================================
# st.markdown("### Simulador Dinâmico de Impacto Legislativo")
with st.expander("ℹ️ Metodologia e Escopo Temporal (Como ler este simulador)", expanded=False):
    st.markdown("""
    **Data de Referência (Baseline):** Os valores base apresentados refletem o estado real do mercado e das pressões urbanas da freguesia no último período validado pelo modelo (**Novembro de 2025**).
    
    **Motor de Simulação:** O algoritmo *Augmented LightGBM* engere **43 variáveis estruturais** em tempo real (incluindo transportes, censos, e atividade económica via satélite). 
    Ao manipular as *Alavancas Políticas* no painel lateral, o simulador recalcula o valor de mercado isolando o choque dessa medida, mantendo as restantes 42 variáveis rigorosamente congeladas na sua realidade de 2025.
    """)

# Caixas de Impacto (Kpis Limpos e Diretos)
col1, col2, col3 = st.columns(3)
with col1:
    st.info("📌 **BASELINE:** Preço estimado pelo modelo para as condições reais de hoje, sem mexer na lei.")
    st.metric(
        label="Preço Preditivo Base", 
        value=f"{preco_base:.2f} €/m²"
    )
with col2:
    st.info("🎯 **CENÁRIO:** Novo preço estimado com as alterações aplicadas no menu lateral.")
    st.metric(
        label="Preço Simulado", 
        value=f"{preco_simulado:.2f} €/m²", 
        delta=f"{impacto_pct:.2f}%"
    )
with col3:
    st.info("💶 **IMPACTO:** Variação isolada atribuída exclusivamente à mudança legislativa.")
    st.metric(label="Variação Absoluta", value=f"{impacto_abs:.2f} €/m²", delta_color="inverse")

# 1. Função para esmagar diferenças de texto (tira acentos, espaços e põe maiúsculas)
def padronizar_nome(nome):
    nome = str(nome).strip().upper()
    nome = unicodedata.normalize('NFKD', nome).encode('ASCII', 'ignore').decode('utf-8')
    return nome

# 2. Preparar matriz com TODAS as freguesias
indices_alvo = list(freguesias_dict.values())
nomes_alvo = list(freguesias_dict.keys())
matriz_mapa = X_test_aug.loc[indices_alvo].copy()

# Guardar os preços base (BASELINE) para calcular o delta depois
precos_baseline_lisboa = np.expm1(lgbm_final.predict(matriz_mapa))

# 3. Injetar a política simulada APENAS na freguesia selecionada
matriz_mapa.loc[idx_alvo, 'Densidade_AL_km2'] = nova_densidade_al
matriz_mapa.loc[idx_alvo, 'Taxa_Conversao_Reabilitacao'] = nova_taxa_reab
matriz_mapa.loc[idx_alvo, 'Stock_12M_ALV_Reabilitacao_Count'] = novo_alv_reab

# 4. Recalcular os preços simulados de toda a cidade
precos_simulados_lisboa = np.expm1(lgbm_final.predict(matriz_mapa))

# Calcular o Delta % para todas as freguesias (Será 0% para todas exceto a alvo)
# Usamos (Novo - Antigo) / Antigo * 100
deltas_percentuais = ((precos_simulados_lisboa - precos_baseline_lisboa) / precos_baseline_lisboa) * 100

# 5. Criar o DataFrame unificado para o mapa
df_mapa = pd.DataFrame({
    'Freguesia_Original': nomes_alvo, 
    'Preço Absoluto': precos_simulados_lisboa,
    'Impacto (Delta %)': deltas_percentuais
})

# Aplicar a padronização das chaves no DataFrame
df_mapa['Chave_Match'] = df_mapa['Freguesia_Original'].apply(padronizar_nome)

# 6. Carregar o GeoJSON original e preparar os Tooltips
geojson_path = BASE_DIR / 'data' / 'processed' / 'lisboa_poligonos_caop.geojson'

with open(geojson_path, 'r', encoding='utf-8') as f:
    mapa_geojson = json.load(f)

# Criar um dicionário para busca rápida de tooltips
tooltip_dict = df_mapa.set_index('Chave_Match')[['Freguesia_Original', 'Preço Absoluto', 'Impacto (Delta %)']].to_dict('index')

for feature in mapa_geojson['features']:
    nome_geo = feature['properties'].get('Freguesia', '')
    chave_m = padronizar_nome(nome_geo)
    feature['properties']['Chave_Match'] = chave_m
    
    # Adicionar os dados reais ao GeoJSON para o Tooltip ler
    if chave_m in tooltip_dict:
        feature['properties']['NomeDisplay'] = tooltip_dict[chave_m]['Freguesia_Original']
        feature['properties']['PrecoDisplay'] = f"{tooltip_dict[chave_m]['Preço Absoluto']:.2f} €/m²"
        feature['properties']['DeltaDisplay'] = f"{tooltip_dict[chave_m]['Impacto (Delta %)']:.2f}%"
    else:
        # Freguesias sem dados (buracos negros)
        feature['properties']['NomeDisplay'] = nome_geo + " (Sem Dados)"
        feature['properties']['PrecoDisplay'] = "N/D"
        feature['properties']['DeltaDisplay'] = "0.00%"


col_mapa, col_insights = st.columns([7, 3], gap="large")
with col_mapa:
    st.subheader("Auditoria Territorial")
    
    # 6. Configurar o Mapa
    mapa_lisboa = folium.Map(location=[38.73, -9.14], zoom_start=12, tiles="CartoDB positron")
    
    # (Verificar se a variável modo_mapa existe, se não, assumir Preço Absoluto)
    if 'modo_mapa' not in locals(): modo_mapa = 'Preço Absoluto Simulado (€/m²)'

    if modo_mapa == 'Preço Absoluto Simulado (€/m²)':
        coluna_alvo = 'Preço Absoluto'
        cor_mapa = 'YlOrRd'
        legenda = 'Preço Preditivo (€/m²)'
    else:
        coluna_alvo = 'Impacto (Delta %)'
        cor_mapa = 'RdYlGn_r' 
        legenda = 'Impacto da Política (Variação %)'

    val_min, val_max = df_mapa[coluna_alvo].min(), df_mapa[coluna_alvo].max()
    escala_dinamica = list(np.linspace(val_min - 1.0, val_max + 1.0, 6)) if val_min == val_max else list(np.linspace(val_min - 0.001, val_max + 0.001, 6))

    choropleth = folium.Choropleth(
        geo_data=mapa_geojson,
        name='choropleth',
        data=df_mapa,
        columns=['Chave_Match', coluna_alvo],
        key_on='feature.properties.Chave_Match',
        fill_color=cor_mapa,
        fill_opacity=0.75,
        line_opacity=0.4,
        legend_name=legenda,
        bins=escala_dinamica,
        highlight=True
    ).add_to(mapa_lisboa)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['NomeDisplay', 'PrecoDisplay', 'DeltaDisplay'],
            aliases=['Freguesia:', 'Preço:', 'Impacto:'],
            style=("background-color: white; color: #333333; font-family: sans-serif; font-size: 13px; padding: 10px;"),
            sticky=True
        )
    )

    chave_atualizacao = f"mapa_{nova_densidade_al}_{nova_taxa_reab}_{novo_alv_reab}_{modo_mapa}"
    # Ajustei a altura para 450 para alinhar melhor com o texto lateral
    st_folium(mapa_lisboa, use_container_width=True, height=450, returned_objects=[], key=chave_atualizacao)

# ==========================================
# 6. MOTOR DE INSIGHTS (Explicação Dinâmica)
# ==========================================

with col_insights:
    st.subheader("🧠 Interpretação")
    
    with st.container():
        insights_gerados = 0
        
        # 1. AL (Métrica Instantânea / Fotografia)
        if nova_densidade_al != val_al:
            if nova_densidade_al < val_al:
                st.success(f"**🏨 Descompressão Turística:** Reduzir a pegada atual para {nova_densidade_al:.0f} AL/km² liberta, no imediato, imóveis do canal turístico para o mercado residencial longo, baixando o teto especulativo.")
            else:
                st.error(f"**📈 Saturação Turística:** Atingir a marca de {nova_densidade_al:.0f} AL/km² sinaliza uma forte canibalização da habitação. O modelo precifica esta escassez residencial com subidas acentuadas.")
            insights_gerados += 1

        # 2. ALVARÁS (Métrica Acumulada Anual / 12 Meses)
        if novo_alv_reab != val_alv_reab:
            if novo_alv_reab > val_alv_reab:
                st.info(f"**🏗️ Choque de Oferta (Ano):** Subir a meta para {novo_alv_reab:.0f} licenças aprovadas num ciclo de 12 meses injeta habitação renovada contínua. Este stock anual ajuda a diluir a pressão da procura.")
            else:
                st.warning(f"**🚧 Estrangulamento (Ano):** Uma meta anual baixa de {novo_alv_reab:.0f} licenças estagna a renovação do parque. Esta retenção do stock a 12 meses força os preços a subir por escassez física.")
            insights_gerados += 1

        # 3. EFICIÊNCIA (Métrica de Fluxo da Máquina Pública)
        if nova_taxa_reab != val_reab_taxa:
            if nova_taxa_reab > val_reab_taxa:
                st.success(f"**⚡ Máquina Municipal:** Operar com {nova_taxa_reab*100:.0f}% de taxa de conversão de pedidos em obra reduz o 'risco-burocrático'. O investimento flui mais rápido para a habitação tangível.")
            else:
                st.error(f"**🐌 Risco-Burocrático:** Uma taxa de conversão baixa ({nova_taxa_reab*100:.0f}%) funciona como um 'imposto invisível'. O mercado repercute os meses de espera da Câmara no preço final pago pelo cidadão.")
            insights_gerados += 1

        if insights_gerados == 0:
            st.info("💡 **Aguardando Simulação:** Ajuste as Alavancas Políticas para visualizar a explicação causal do modelo.")

    st.markdown(
        f"<div style='text-align: right; color: #94A3B8; font-size: 0.75rem; margin-top: 2rem;'>"
        f"Dados: Censos 2021 + Séries Temporais (Nov 2025)</div>", 
        unsafe_allow_html=True
    )