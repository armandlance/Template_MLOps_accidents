"""
Dashboard Streamlit pour visualiser les performances et prédictions
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os

# Configuration de la page
st.set_page_config(
    page_title="Road Accident MLOps Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Titre
st.title("🚗 Road Accident Prediction Dashboard")

# Sidebar pour la navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "🔮 Prédiction", "📊 Performances", "⚠️ Monitoring"]
)

# Configuration API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page Home
if page == "🏠 Home":
    st.header("Bienvenue sur le Dashboard MLOps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Services", "3/3", "Running")
    
    with col2:
        st.metric("Modèle", "v1.0", "Active")
    
    with col3:
        st.metric("Accuracy", "0.85", "+2%")
    
    # Health check
    st.subheader("🏥 Health Check")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is healthy")
            st.json(response.json())
        else:
            st.error("❌ API is not responding")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")

# Page Prédiction
elif page == "🔮 Prédiction":
    st.header("Faire une prédiction")
    
    st.info("📝 Entrez les caractéristiques de l'accident pour obtenir une prédiction")
    
    # Formulaire de prédiction
    with st.form("prediction_form"):
        st.subheader("Caractéristiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.number_input("Feature 1", value=0.5)
            feature2 = st.number_input("Feature 2", value=1.2)
            feature3 = st.number_input("Feature 3", value=0.8)
        
        with col2:
            feature4 = st.number_input("Feature 4", value=0.3)
            feature5 = st.number_input("Feature 5", value=1.5)
            feature6 = st.number_input("Feature 6", value=0.6)
        
        submitted = st.form_submit_button("🎯 Prédire")
        
        if submitted:
            # Préparer les données
            features = {
                "feature1": feature1,
                "feature2": feature2,
                "feature3": feature3,
                "feature4": feature4,
                "feature5": feature5,
                "feature6": feature6
            }
            
            payload = {"features": features}
            
            # Faire la prédiction
            try:
                with st.spinner("Prédiction en cours..."):
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=10
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Prédiction réussie!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Prédiction",
                            f"Classe {result['prediction']}"
                        )
                    
                    with col2:
                        st.metric(
                            "Probabilité",
                            f"{result['probability']:.2%}"
                        )
                    
                    # Afficher les détails
                    with st.expander("📄 Détails de la requête"):
                        st.json(payload)
                    
                    with st.expander("📊 Réponse complète"):
                        st.json(result)
                
                else:
                    st.error(f"❌ Erreur: {response.status_code}")
                    st.json(response.json())
            
            except Exception as e:
                st.error(f"❌ Erreur de connexion: {str(e)}")

# Page Performances
elif page == "📊 Performances":
    st.header("Performances du modèle")
    
    # Exemple de métriques (à adapter avec tes vraies données)
    metrics_data = {
        'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Valeur': [0.85, 0.82, 0.88, 0.85],
        'Train': [0.87, 0.84, 0.90, 0.87]
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    # Affichage des métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "85%", "2%")
    with col2:
        st.metric("Precision", "82%", "1%")
    with col3:
        st.metric("Recall", "88%", "3%")
    with col4:
        st.metric("F1-Score", "85%", "2%")
    
    # Graphique
    fig = go.Figure(data=[
        go.Bar(name='Test', x=df_metrics['Métrique'], y=df_metrics['Valeur']),
        go.Bar(name='Train', x=df_metrics['Métrique'], y=df_metrics['Train'])
    ])
    
    fig.update_layout(
        title="Comparaison Train vs Test",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Evolution des performances
    st.subheader("📈 Évolution des performances")
    
    # Exemple de données temporelles
    dates = pd.date_range('2024-01-01', periods=10, freq='W')
    performance_history = {
        'Date': dates,
        'Accuracy': [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.85, 0.85, 0.86, 0.85]
    }
    df_history = pd.DataFrame(performance_history)
    
    fig2 = px.line(df_history, x='Date', y='Accuracy', 
                   title='Évolution de l\'Accuracy',
                   markers=True)
    
    st.plotly_chart(fig2, use_container_width=True)

# Page Monitoring
elif page == "⚠️ Monitoring":
    st.header("Monitoring et Alertes")
    
    st.subheader("🔍 Data Drift Detection")
    
    # Exemple de résultats de drift
    drift_data = {
        'Feature': ['feature1', 'feature2', 'feature3', 'feature4'],
        'P-Value': [0.08, 0.03, 0.15, 0.02],
        'Status': ['OK', 'DRIFT', 'OK', 'DRIFT']
    }
    df_drift = pd.DataFrame(drift_data)
    
    # Colorier selon le status
    def highlight_drift(row):
        if row['Status'] == 'DRIFT':
            return ['background-color: #ffcccc'] * len(row)
        else:
            return ['background-color: #ccffcc'] * len(row)
    
    st.dataframe(
        df_drift.style.apply(highlight_drift, axis=1),
        use_container_width=True
    )
    
    # Alertes
    st.subheader("🚨 Alertes récentes")
    
    if os.path.exists('./logs/drift_alerts.json'):
        with open('./logs/drift_alerts.json', 'r') as f:
            alerts = [json.loads(line) for line in f]
        
        for alert in alerts[-5:]:  # 5 dernières alertes
            with st.expander(f"⚠️ Alerte - {alert['timestamp']}"):
                st.json(alert)
    else:
        st.info("Aucune alerte enregistrée")
    
    # Actions recommandées
    st.subheader("💡 Actions recommandées")
    
    if any(df_drift['Status'] == 'DRIFT'):
        st.warning("⚠️ Data drift détecté! Actions recommandées:")
        st.markdown("""
        - 🔄 Réentraîner le modèle avec les nouvelles données
        - 📊 Analyser les features qui dérivent
        - 🔍 Vérifier la qualité des données récentes
        - 📧 Notifier l'équipe data science
        """)
    else:
        st.success("✅ Pas de drift détecté. Le modèle fonctionne normalement.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Road Accident MLOps**
    
    Version: 1.0.0
    
    [Documentation](https://github.com/your-repo)
    """
)