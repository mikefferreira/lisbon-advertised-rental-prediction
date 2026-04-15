# Lisbon Advertised Rental Market Prediction: An Augmented Approach

Este repositório contém a infraestrutura de dados e modelos da tese.

## Objetivo do Projeto
Prever o preço médio de arrendamento por $m^2$ nas freguesias de Lisboa através de um modelo **Augmented**. A inovação reside na integração de fontes heterogéneas:
* **Mercado:** Dados de anúncios (Idealista).
* **Urbanismo:** Licenciamentos e reabilitação urbana (RJUE).
* **Turismo:** Densidade de Alojamento Local (RNAL).
* **Remote Sensing:** Luminosidade noturna via satélite (VIIRS).
* **Demografia:** Dados estruturais (Censos 2011/2021).

## 📂 Estrutura do Repositório
* `notebooks/`: Pipeline completo de Processamento, Feature Engineering e Modelação.
* `reports/plots/`: Visualizações de impacto (SHAP values, ICE plots e métricas de performance).
* `src/`: Scripts auxiliares e o simulador interativo de preços.

## Ordem de Execução (Pipeline)
Para replicar os resultados, os notebooks devem ser executados na seguinte ordem:
1. Data Acquisition: Execução dos scripts de recolha de dados do Idealista.
2. `VAR_*.ipynb`: Processamento individual de cada fonte de dados.
3. `final_dataset.ipynb`: Faz o merge de todas as fontes de dados (processed/) numa única tabela mestre.
4. `CRISP-ML.ipynb`: Treino do modelo LightGBM, validação cruzada e análise de importância de variáveis (SHAP).
5. `app_simulator.py`: Interface interativa para prever preços e simular cenários.

## ⚠️ Nota sobre os Dados
Por motivos de confidencialidade e volume, a pasta `data/` está excluída deste repositório. Para execução, os datasets brutos devem ser colocados em `data/raw/` conforme as instruções nos notebooks.
