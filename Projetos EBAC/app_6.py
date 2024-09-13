import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import os

from pycaret.classification import setup, pull, load_model, compare_models, predict_model, models, save_model, plot_model, finalize_model
from pycaret.utils.generic import check_metric

# Verifica se o arquivo 'df.csv' existe e carrega os dados
if os.path.exists('df.csv'):
    df = pd.read_csv('df.csv', index_col=False)

st.set_page_config(layout='wide',
                   page_title='ML app',
                   page_icon='ml_icon.png',
                   menu_items={'About': 'https://github.com/edudatalytics'})

with st.sidebar:
    st.title('Machine Learning App')
    choice = st.radio('Aplicação:', [
        'Upload dos dados', 'Machine Learning',
        'Visualização', 'Download'
    ])
    st.info('''Este aplicativo permite que você carregue dados e crie um modelo de 
    Machine Learning de classificação automatizado com visualizações de
    performance e download do modelo e resultados. Carregue os dados e veja a mágica acontecer!''')

file = None

if choice == 'Upload dos dados':
    st.title('Faça o upload dos seus dados!')
    st.markdown('---')
    file = st.file_uploader('Upload dos dados em formato *.csv* ou *.ftr*:')

    if file:
        try:
            # Verifica a extensão do arquivo e carrega-o adequadamente
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.ftr'):
                df = pd.read_feather(file)
            else:
                st.error("Formato de arquivo não suportado. Por favor, faça o upload de um arquivo .csv ou .feather.")
                df = None
            
            if df is not None:
                # Remove valores nulos
                df = df.dropna()
                df.to_csv('df.csv', index=False)
                st.markdown('Amostra do dataframe:')
                st.dataframe(df.head(10))

        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")

if choice == 'Machine Learning':

    st.title('Criando o Modelo Para Seus Dados!')
    st.markdown('---')

    if df is not None:
        alvo = st.selectbox('Selecione a variável alvo:', df.columns)

        data = df.sample(frac=0.75, random_state=42)
        data_val = df.drop(data.index)
        data.reset_index(inplace=True, drop=True)

        st.write('''Aqui o melhor modelo para prever sua variável alvo será selecionado e treinado
        com seus dados!''')

        if st.button('Treinar Modelo'):

            # Configura o ambiente de PyCaret
            setup(data=data, target=alvo)  # Remover parâmetros não suportados
            setup_df = pull()

            st.write('## Lista de modelos disponíveis:')
            st.write(models())
            st.markdown('---')

            st.info('Configuração do experimento de Classificação:')
            st.dataframe(setup_df)

            best_model = compare_models(sort='AUC')
            st.info('Modelo Final:')
            st.write(best_model)
            st.write('Por questões de tempo de processamento do Streamlit, a etapa de tunning do modelo não está ativada.')

            st.info('Métricas na base de validação:')

            data_val_pred = predict_model(best_model, data=data_val)

            st.write('Acurácia:')
            accuracy = check_metric(data_val_pred[alvo], data_val_pred['prediction_label'], metric='Accuracy')
            st.write(accuracy)
            st.write('AUC:')
            auc = check_metric(data_val_pred[alvo], data_val_pred['prediction_label'], metric='AUC')
            st.write(auc)

            save_model(best_model, 'Model')
    else:
        st.error('Por favor, faça o upload dos dados antes de treinar o modelo.')

if choice == 'Visualização':

    st.title('Visualização da Performance do Modelo')
    st.markdown('---')

    try:
        modelo = load_model('C:\\Users\\User\\Desktop\\model_final_pycaret')

        st.write('## Features Mais Relevantes')
        fig_feat = plot_model(modelo, plot='feature')
        st.pyplot(fig_feat)
        st.info('Feature Importance indica qual variável é mais influente para a previsão.')

        st.write('## Curva ROC-AUC')
        fig_auc = plot_model(modelo, plot='auc')
        st.pyplot(fig_auc)
        st.info('Curva ROC-AUC é uma importante medida de desempenho do modelo, variando de 0 e 1.')

        st.write('## Matriz de Confusão')
        fig_mc = plot_model(modelo, plot='confusion_matrix')
        st.pyplot(fig_mc)
        st.info('A Matriz de Confusão cria uma tabela cruzada entre os valores reais no eixo *y* e os valores preditos no eixo *x*.')
    except Exception as e:
        st.error(f"Erro ao carregar o modelo para visualização: {e}")

if choice == 'Download':

    st.title('Downloads')
    st.markdown('---')

    st.info('Treinando o modelo final em toda base de dados...')
    
    try:
        modelo = load_model('Model')
        modelo_final = finalize_model(modelo)
        save_model(modelo_final, 'Final_Model')

        st.info('Dados classificados como evento:')
        df_pred = predict_model(modelo_final, df)
        st.dataframe(df_pred[df_pred['prediction_label'] == 1])

        st.info('Download dos dados com previsão:')
        csv = df_pred.to_csv(index=False).encode()
        st.download_button(label='.csv',
                           data=csv,
                           file_name='data_prev.csv',
                           mime='text/csv')
        
        st.info('Download do modelo:')
        with open('Final_Model.pkl', 'rb') as f:
            st.download_button(label='.pkl',
                               data=f,
                               file_name='modelo_final.pkl')
    except Exception as e:
        st.error(f"Erro ao salvar o modelo ou gerar o download: {e}")
