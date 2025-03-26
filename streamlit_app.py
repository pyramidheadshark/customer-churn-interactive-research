import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import shap
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide", page_title="Дешборд Анализа Оттока")

BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Churn_Modelling.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_churn_model.joblib')

X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train_processed.csv')
X_TEST_PATH = os.path.join(DATA_DIR, 'X_test_processed.csv')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.csv')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.csv')


@st.cache_data
def load_processed_data():
    """Загружает обработанные обучающие и тестовые данные."""
    try:
        X_train = pd.read_csv(X_TRAIN_PATH)
        X_test = pd.read_csv(X_TEST_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).squeeze('columns')
        y_test = pd.read_csv(Y_TEST_PATH).squeeze('columns')

        if isinstance(y_train, pd.DataFrame): y_train = y_train[y_train.columns[0]]
        if isinstance(y_test, pd.DataFrame): y_test = y_test[y_test.columns[0]]
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки обработанных данных: {e}. Пожалуйста, проверьте пути к файлам.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке обработанных данных: {e}")
        return None, None, None, None

@st.cache_data
def load_original_data():
    """Загружает исходные, необработанные данные для EDA."""
    try:
        df_original = pd.read_csv(ORIGINAL_DATA_PATH)
        df_original = df_original.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'HasCrCard'], errors='ignore')
        return df_original
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки исходных данных: {e}. Пожалуйста, проверьте пути к файлам.")
        return None
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке исходных данных: {e}")
        return None

@st.cache_resource
def load_model():
    """Загружает предобученную модель LightGBM."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки модели: {e}. Убедитесь, что файл модели существует по пути '{MODEL_PATH}'.")
        return None
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке модели: {e}")
        return None

@st.cache_resource
def get_explainer(_model, _X_train):
    """Создает объект SHAP explainer для модели."""
    if _model is None or _X_train is None:
        return None
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception as e:
        st.error(f"Ошибка создания SHAP explainer: {e}")
        return None

@st.cache_data
def calculate_shap_values(_explainer, _X_test):
    """Рассчитывает значения SHAP для тестового набора."""
    if _explainer is None or _X_test is None:
        return None, None
    try:
        st.info("Расчет значений SHAP для тестового набора (это может занять некоторое время)...")
        shap_values_output = _explainer.shap_values(_X_test)

        if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
            shap_values_class1 = shap_values_output[1]
        else:
             shap_values_class1 = shap_values_output

        expected_value = _explainer.expected_value
        if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
             base_value = expected_value[1]
        else:
             base_value = expected_value

        st.success("Значения SHAP рассчитаны.")
        return shap_values_class1, base_value
    except Exception as e:
        st.error(f"Ошибка расчета значений SHAP: {e}")
        return None, None

X_train, X_test, y_train, y_test = load_processed_data()
df_original = load_original_data()
model = load_model()

if X_train is not None and X_test is not None and y_test is not None and model is not None:
    explainer = get_explainer(model, X_train)
    shap_values, base_value = calculate_shap_values(explainer, X_test)

    st.title("📈 Интерактивный Дешборд Анализа Оттока Клиентов")
    st.markdown("Исследование факторов, влияющих на отток клиентов банка, с использованием модели LightGBM и интерпретации SHAP.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 **Обзор Модели**",
        "🌍 **Глобальная Интерпретация**",
        "👤 **Анализ Клиента**",
        "🔬 **EDA**"
    ])

    with tab1:
        st.header("Производительность Модели LightGBM (на Тестовых Данных)")

        if y_test is not None:
            col1, col2, col3 = st.columns(3)
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)

                from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix

                roc_auc = roc_auc_score(y_test, y_pred_proba)
                pr_auc = average_precision_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)

                col1.metric("ROC AUC", f"{roc_auc:.3f}", help="Способность модели ранжировать клиентов по вероятности оттока.")
                col1.metric("PR AUC", f"{pr_auc:.3f}", help="Площадь под кривой Precision-Recall, важна при дисбалансе.")
                col2.metric("F1 Score (Класс 1)", f"{f1:.3f}", help="Гармоническое среднее Precision и Recall для класса Отток.")
                col2.metric("Recall (Класс 1)", f"{recall:.3f}", help="Доля реальных отточных клиентов, которых модель верно определила.")
                col3.metric("Precision (Класс 1)", f"{precision:.3f}", help="Доля клиентов, предсказанных как отточные, которые действительно ушли.")
                col3.metric("Accuracy", f"{model.score(X_test, y_test):.3f}", help="Общая доля верных предсказаний (менее показательна при дисбалансе).")


                st.subheader("Матрица Ошибок (Confusion Matrix)")
                cm = confusion_matrix(y_test, y_pred)
                cm_labels = ['Лоялен (0)', 'Отток (1)']
                fig_cm = ff.create_annotated_heatmap(
                    z=cm, x=cm_labels, y=cm_labels, colorscale='Blues',
                    annotation_text=[[str(y) for y in x] for x in cm]
                )
                fig_cm.update_layout(
                    xaxis_title="Предсказанный класс",
                    yaxis_title="Истинный класс"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            except Exception as e:
                st.error(f"Не удалось рассчитать или отобразить метрики: {e}")
        else:
            st.warning("Тестовые метки (y_test) не загружены. Невозможно отобразить производительность модели.")

    with tab2:
        st.header("🌍 Какие Факторы Влияют на Отток в Целом?")
        if shap_values is not None and not X_test.empty:
            try:
                plt.style.use('default')

                st.subheader("Средняя Важность Признаков (SHAP Bar Plot)")
                fig_bar, ax_bar = plt.subplots()
                shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
                st.pyplot(fig_bar)
                plt.close(fig_bar)
                st.caption("Показывает среднее абсолютное влияние каждого признака на предсказание модели по всем клиентам.")


                st.subheader("Влияние Значений Признаков (SHAP Dot Plot)")
                fig_dot, ax_dot = plt.subplots()
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(fig_dot)
                plt.close(fig_dot)
                st.caption("""
                    Каждая точка - влияние признака для одного клиента.
                    *   **Положение по горизонтали:** Влияние на вероятность оттока (вправо - повышает, влево - понижает).
                    *   **Цвет точки:** Значение признака (красный - высокое, синий - низкое).
                    Признаки упорядочены по суммарной важности сверху вниз.
                """)

                st.subheader("Зависимость Предсказания от Значения Признака")
                feature_options = X_test.columns.tolist()
                selected_feature = st.selectbox("Выберите признак для анализа зависимости:", feature_options, index=feature_options.index('Age') if 'Age' in feature_options else 0)

                if selected_feature:
                    fig_dep, ax_dep = plt.subplots()
                    shap.dependence_plot(selected_feature, shap_values, X_test, interaction_index="auto", show=False, ax=ax_dep)
                    st.pyplot(fig_dep)
                    plt.close(fig_dep)
                    st.caption(f"Показывает, как меняется вклад признака '{selected_feature}' в предсказание оттока при изменении его значения. Цвет точек может указывать на взаимодействие с другим признаком.")

            except ImportError:
                 st.warning("Пожалуйста, убедитесь, что matplotlib установлен для отображения SHAP графиков.")
            except Exception as e:
                 st.error(f"Не удалось отобразить SHAP графики: {e}")

        else:
            st.warning("SHAP значения не рассчитаны или тестовые данные не загружены.")

    with tab3:
        st.header("👤 Почему Модель Приняла Такое Решение по Конкретному Клиенту?")

        if shap_values is not None and base_value is not None and not X_test.empty and y_test is not None and explainer is not None:
            max_index = len(X_test) - 1
            customer_index = st.number_input(
                f"Выберите индекс клиента из тестовой выборки (от 0 до {max_index}):",
                min_value=0, max_value=max_index, value=0, step=1
            )

            if customer_index is not None and 0 <= customer_index <= max_index:
                st.subheader(f"Анализ для Клиента с Индексом {customer_index}")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Характеристики клиента:**")
                    st.dataframe(X_test.iloc[[customer_index]].T.rename(columns={customer_index: 'Значение'}))
                with col2:
                    actual_status = "Отток" if y_test.iloc[customer_index] == 1 else "Лоялен"
                    pred_proba = model.predict_proba(X_test.iloc[[customer_index]])[0, 1]
                    pred_class = model.predict(X_test.iloc[[customer_index]])[0]
                    st.metric("Реальный статус:", actual_status)
                    st.metric("Предсказанная вероятность оттока:", f"{pred_proba:.2%}")
                    st.metric("Предсказанный класс:", "Отток" if pred_class == 1 else "Лоялен")


                st.subheader("Вклад Факторов в Предсказание (SHAP Waterfall Plot)")
                st.markdown("Визуализация показывает, как вклад отдельных признаков изменяет предсказание от среднего значения (`E[f(X)]`) до финального (`f(x)`) для этого клиента.")

                try:
                    explanation = shap.Explanation(
                        values=shap_values[customer_index,:],
                        base_values=base_value,
                        data=X_test.iloc[customer_index,:].values,
                        feature_names=X_test.columns.tolist()
                    )

                    fig_waterfall, ax_waterfall = plt.subplots()
                    shap.waterfall_plot(explanation, max_display=15, show=False)
                    st.pyplot(fig_waterfall, bbox_inches='tight')
                    plt.close(fig_waterfall)

                    st.caption("""
                        *   **E[f(X)] ({:.3f}):** Среднее предсказание модели (логарифм шансов оттока).
                        *   **Красные стрелки:** Факторы/признаки, увеличивающие предсказание оттока.
                        *   **Синие стрелки:** Факторы/признаки, уменьшающие предсказание оттока.
                        *   **f(x) ({:.3f}):** Финальное предсказание модели (логарифм шансов) для этого клиента.
                    """.format(base_value, explanation.values.sum() + base_value))

                except Exception as e:
                    st.error(f"Не удалось создать Waterfall Plot: {e}")


                with st.expander("Показать таблицу вклада признаков"):
                    shap_df = pd.DataFrame({
                        'Признак': X_test.columns,
                        'Значение': X_test.iloc[customer_index].values,
                        'SHAP Value': shap_values[customer_index,:]
                    })
                    st.dataframe(shap_df.sort_values(by='SHAP Value', key=abs, ascending=False))

            else:
                 st.warning("Пожалуйста, введите корректный индекс клиента.")
        else:
            missing = []
            if shap_values is None: missing.append("значения SHAP")
            if base_value is None: missing.append("базовое значение")
            if X_test.empty: missing.append("тестовые данные (X_test)")
            if y_test is None: missing.append("тестовые метки (y_test)")
            if explainer is None: missing.append("SHAP explainer")
            st.warning(f"Невозможно выполнить локальную интерпретацию. Отсутствуют: {', '.join(missing)}.")

    with tab4:
        st.header("🔬 Исследование Исходных Данных")

        if df_original is not None:
            st.markdown("Анализ распределения признаков до предобработки.")

            feature_to_plot = st.selectbox(
                "Выберите признак для визуализации:",
                df_original.columns.drop('Exited', errors='ignore'),
                index=df_original.columns.get_loc('Age') if 'Age' in df_original.columns else 0
            )

            if feature_to_plot:
                col_eda1, col_eda2 = st.columns([1, 1])

                with col_eda1:
                     st.write(f"**Распределение признака '{feature_to_plot}'**")
                     if pd.api.types.is_numeric_dtype(df_original[feature_to_plot]) and df_original[feature_to_plot].nunique() > 15:
                         fig_eda = px.histogram(df_original, x=feature_to_plot, marginal="box", title=f"Распределение {feature_to_plot}")
                     else:
                         fig_eda = px.bar(df_original[feature_to_plot].value_counts().reset_index(),
                                         x=feature_to_plot, y='count', title=f"Распределение {feature_to_plot}",
                                         text_auto=True)
                         fig_eda.update_layout(xaxis={'categoryorder':'total descending'})
                     st.plotly_chart(fig_eda, use_container_width=True)

                with col_eda2:
                    if 'Exited' in df_original.columns:
                        st.write(f"**Распределение '{feature_to_plot}' в разрезе оттока ('Exited')**")
                        if pd.api.types.is_numeric_dtype(df_original[feature_to_plot]) and df_original[feature_to_plot].nunique() > 15:
                             fig_eda_churn = px.histogram(df_original, x=feature_to_plot, color='Exited',
                                                        marginal="box", barmode='overlay', opacity=0.7,
                                                        title=f"{feature_to_plot} по Статусу Оттока")
                        else:
                              df_grouped = df_original.groupby([feature_to_plot, 'Exited']).size().reset_index(name='count')
                              fig_eda_churn = px.bar(df_grouped, x=feature_to_plot, y='count', color='Exited',
                                                    barmode='group', title=f"{feature_to_plot} по Статусу Оттока",
                                                    text_auto=True)
                              fig_eda_churn.update_layout(xaxis={'categoryorder':'total descending'})

                        st.plotly_chart(fig_eda_churn, use_container_width=True)
                    else:
                        st.info("Колонка 'Exited' не найдена в исходных данных для анализа в разрезе оттока.")

        else:
            st.warning("Исходные данные не загружены. Невозможно отобразить EDA.")

else:
    st.error("Не удалось загрузить необходимые данные или модель. Пожалуйста, проверьте конфигурацию и пути к файлам.")

st.sidebar.title("О Проекте")
st.sidebar.info(
    """
    **Дешборд для Анализа Оттока Клиентов**

    **Цель:** Интерактивно исследовать результаты модели машинного обучения (LightGBM), предсказывающей отток клиентов, и понять ключевые факторы, влияющие на него.

    **Данные:** Набор данных о клиентах банка.
    **Модель:** LightGBM, оптимизированный с помощью Optuna.
    **Интерпретация:** SHAP (SHapley Additive exPlanations).

    **Вкладки:**
    - **Обзор Модели:** Основные метрики качества.
    - **Глобальная Интерпретация:** Важность признаков в целом.
    - **Анализ Клиента:** Объяснение предсказания для конкретного клиента.
    - **EDA:** Исследование исходных данных.
    """
)