import time
import streamlit as st
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

yeast = fetch_ucirepo(id=110)
X = yeast.data.features
y = yeast.data.targets

binary_target = y['localization_site'].apply(lambda x: 1 if x in ['MIT', 'CYT'] else 0)
X_target = X.copy()
X_target['binary_target'] = binary_target
st.title("YeastFinder 🍂 ")
st.write("Описание: предсказания локализации белков в клетках дрожжей на основе их характеристик.")

with st.expander('Data'):
    st.write('X')
    st.dataframe(X)

    st.write('y')
    st.dataframe(binary_target)

with st.sidebar:
    dark_mode = st.toggle("Dark theme")
    if dark_mode:
        st.write("Dark mode is enabled!")
    else:
        st.write("Light mode is enabled!")
    st.header("Input features: ")
    mcg = st.slider('mcg', 0.0, 1.0, 0.3)
    gvh = st.slider('gvh', 0.0, 1.0, 0.4)
    alm = st.slider('alm', 0.0, 1.0, 0.2)
    mit = st.slider('mit', 0.0, 1.0, 0.5)
    erl = st.slider('erl', 0.0, 1.0, 0.4)
    pox = st.slider('pox', 0.0, 1.0, 0.6)
    vac = st.slider('vac', 0.0, 1.0, 0.9)
    nuc = st.slider('nuc', 0.0, 1.0, 0.3)
    models = st.multiselect('Models ', ('Logistic Regression', 'KNN', 'Decision Tree'))

st.subheader('Data Visualization')

fig = px.scatter_3d(
    X_target,
    x='gvh', y='alm', z='mit',
    color='binary_target',
    title='Три важных признака',
    symbol='binary_target',
    size_max=2,

)
fig.update_traces(
    marker=dict(
        size=4,
        opacity=0.6,
        line=dict(width=1, color='black')
    )
)
fig.update_layout(
    scene=dict(
        xaxis_title='gvh',
        yaxis_title='alm',
        zaxis_title='mit'
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

df = pd.concat([X, binary_target], axis=1)

corr = df.corr()
plt.figure(figsize=(15, 7))
sns.heatmap(corr, cmap='mako', annot=True)
st.pyplot(plt)

input_data = {
    'mcg': mcg,
    'gvh': gvh,
    'alm': alm,
    'mit': mit,
    'erl': erl,
    'pox': pox,
    'vac': vac,
    'nuc': nuc,
}

X_train, X_test, y_train, y_test = train_test_split(X, binary_target, test_size=0.2, random_state=42)

st.plotly_chart(fig, use_container_width=True)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models_dict = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
}
input_df = pd.DataFrame(input_data, index=[0])
input_scaled = scaler.transform(input_df)

with st.expander('Input feature'):
    st.write('*Input plugins*')
    st.dataframe(input_df)
    st.write('*Scaled data*')
    st.dataframe(input_scaled)

for select_model in models:
    model = models_dict[select_model]
    model.fit(X_train_scaled, y_train)

    model_proba = model.predict_proba(X_test_scaled)[:, 1]
    model_roc_auc = roc_auc_score(y_test, model_proba)
    fpr, tpr, thesholds = roc_curve(y_test, model_proba)

    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, label=f'{select_model} AUC = {model_roc_auc}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc-Auc for 3 models')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    prediction = model.predict(input_scaled)
    pred_proba = model.predict_proba(input_scaled)

    df_prediction_proba = pd.DataFrame(pred_proba, columns=['Not MIT/CYT (0)', 'MIT/CYT (1)'])

    if st.button(f'Predicted {select_model}'):
        with st.spinner('Have a good day :)! Predicting...'):

            progress_bar = st.progress(0)
            progress_text = st.empty()
            for i in range(101):
                time.sleep(0.1)
                progress_bar.progress(i)
                progress_text.text(f'Progress: {i}%')

            st.subheader(f'Prediction for {select_model}')
            st.dataframe(
                df_prediction_proba,
                column_config={
                    'Not MIT/CYT (0)': st.column_config.ProgressColumn(
                        'Not MIT/CYT (0)',
                        format='%.2f',
                        width='medium',
                        min_value=0,
                        max_value=1,
                    ),
                    'MIT/CYT (1)': st.column_config.ProgressColumn(
                        'MIT/CYT (1)',
                        format='%.2f',
                        width='medium',
                        min_value=0,
                        max_value=1,
                    )
                },
                hide_index=True
            )

            result_label = 'MIT/CYT' if prediction == 1 else 'Other'
            st.success(f"Predicted category: {result_label}")
