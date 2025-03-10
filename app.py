import streamlit as st
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import shutil
import tempfile
from dotenv import load_dotenv
from google.oauth2 import service_account

# Importa le funzioni dal file principale
# Assicurati che il file principale si chiami main.py e sia nella stessa directory
from main import analyze_text_sentiment, analyze_face_expression, transcribe_audio

st.set_page_config(
    page_title="Analizzatore di Sentiment e Ironia", 
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funzione per caricare le credenziali dal file .env
def load_credentials_from_env():
    """Safely load Google Cloud credentials from Streamlit secrets or environment"""
    import streamlit as st
    import os
    import json
    import tempfile
    
    credentials_dict = {}
    
    # First try accessing secrets directly through Streamlit
    try:
        # Try dictionary-style access first
        credentials_dict = {
            "type": st.secrets["GOOGLE_TYPE"],
            "project_id": st.secrets["GOOGLE_PROJECT_ID"],
            "private_key_id": st.secrets["GOOGLE_PRIVATE_KEY_ID"],
            "private_key": st.secrets["GOOGLE_PRIVATE_KEY"].replace("\\n", "\n"),
            "client_email": st.secrets["GOOGLE_CLIENT_EMAIL"],
            "client_id": st.secrets["GOOGLE_CLIENT_ID"],
            "auth_uri": st.secrets["GOOGLE_AUTH_URI"],
            "token_uri": st.secrets["GOOGLE_TOKEN_URI"],
            "auth_provider_x509_cert_url": st.secrets["GOOGLE_AUTH_PROVIDER_X509_CERT_URL"],
            "client_x509_cert_url": st.secrets["GOOGLE_CLIENT_X509_CERT_URL"],
            "universe_domain": st.secrets.get("GOOGLE_UNIVERSE_DOMAIN", "googleapis.com")
        }
        st.sidebar.success("✅ Credenziali caricate da secrets")
    except Exception:
        # If that fails, load from .env or environment variables
        try:
            # Try to load from .env file
            try:
                from dotenv import load_dotenv
                app_dir = os.path.dirname(__file__)
                env_path = os.path.join(app_dir, '.env')
                load_dotenv(env_path)
            except Exception:
                pass  # Continue with environment variables even if .env loading fails
            
            # Get values with safe defaults
            private_key = os.getenv("GOOGLE_PRIVATE_KEY")
            if private_key:
                private_key = private_key.replace("\\n", "\n")
                
            credentials_dict = {
                "type": os.getenv("GOOGLE_TYPE", ""),
                "project_id": os.getenv("GOOGLE_PROJECT_ID", ""),
                "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID", ""),
                "private_key": private_key,
                "client_email": os.getenv("GOOGLE_CLIENT_EMAIL", ""),
                "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
                "auth_uri": os.getenv("GOOGLE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": os.getenv("GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_X509_CERT_URL", 
                                                          "https://www.googleapis.com/oauth2/v1/certs"),
                "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_CERT_URL", ""),
                "universe_domain": os.getenv("GOOGLE_UNIVERSE_DOMAIN", "googleapis.com")
            }
            st.sidebar.info("ℹ️ Usando variabili d'ambiente")
        except Exception as e:
            st.sidebar.error(f"❌ Errore nel caricamento delle credenziali: {str(e)}")
            return None
    
    # Verify required keys exist and have values
    required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email"]
    missing_keys = [key for key in required_keys if not credentials_dict.get(key)]
    
    if missing_keys:
        st.sidebar.error(f"⚠️ Credenziali mancanti: {', '.join(missing_keys)}")
        st.sidebar.info("Aggiungi le credenziali nelle impostazioni Streamlit Cloud > Settings > Secrets")
        return None
    
    # Create a temporary file with credentials
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            json.dump(credentials_dict, temp_file)
            temp_path = temp_file.name
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        return temp_path
    except Exception as e:
        st.sidebar.error(f"❌ Errore nella creazione del file temporaneo: {str(e)}")
        return None

# Funzione per verificare la presenza dei file predefiniti
def check_default_files():
    app_dir = os.path.dirname(__file__)
    default_image = os.path.join(app_dir, "test.jpg")
    default_audio = os.path.join(app_dir, "test.wav")
    
    image_exists = os.path.exists(default_image)
    audio_exists = os.path.exists(default_audio)
    
    return {
        "image": {"path": default_image, "exists": image_exists},
        "audio": {"path": default_audio, "exists": audio_exists}
    }

# Funzione per salvare temporaneamente un file caricato
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Crea una directory temporanea se non esiste
        temp_dir = tempfile.gettempdir()
        # Salva il file
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

# Funzione per visualizzare il sentiment score con un gauge chart
def plot_sentiment_gauge(score, title="Sentiment Score"):
    fig, ax = plt.subplots(figsize=(4, 0.8), subplot_kw=dict(polar=True))
    
    # Converti il sentiment score da [-1, 1] a [0, 1]
    normalized_score = (score + 1) / 2
    
    # Colori per il gauge (rosso per negativo, verde per positivo)
    color = plt.cm.RdYlGn(normalized_score)
    
    # Disegna il gauge
    pos = 0.5  # Center position
    bar_height = 0.1
    ax.bar(
        x=np.pi, 
        height=bar_height,
        width=2*np.pi,
        bottom=pos-bar_height/2,
        color='lightgrey',
        alpha=0.5
    )
    
    # Barra del sentiment
    ax.bar(
        x=np.pi, 
        height=bar_height,
        width=2*np.pi*normalized_score,
        bottom=pos-bar_height/2,
        color=color,
        alpha=0.8
    )
    
    # Personalizzazione del plot
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Etichette
    plt.annotate(
        f"{score:.2f}",
        xy=(0.5, 0.5),
        xycoords='figure fraction',
        ha='center',
        va='center',
        fontsize=14
    )
    
    # Rimuovi i bordi e le linee delle griglie
    ax.spines['polar'].set_visible(False)
    
    plt.tight_layout()
    
    return fig

# Funzione per visualizzare le emozioni dalle espressioni facciali
def plot_emotions(emotions):
    if "error" in emotions:
        return None
    
    # Mapping dei valori di likelihood a numeri
    likelihood_mapping = {
        "UNKNOWN": 0,
        "VERY_UNLIKELY": 1,
        "UNLIKELY": 2,
        "POSSIBLE": 3,
        "LIKELY": 4,
        "VERY_LIKELY": 5
    }
    
    # Estrai i valori numerici delle emozioni
    emotion_values = {}
    for emotion, likelihood in emotions.items():
        if emotion != "detection_confidence":
            if likelihood in likelihood_mapping:
                emotion_values[emotion] = likelihood_mapping[likelihood]
    
    if not emotion_values:
        return None
    
    # Crea il grafico a barre
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Assegna colori alle emozioni
    colors = {
        "joy": "#FFD700",      # Oro
        "sorrow": "#4682B4",   # Blu acciaio
        "anger": "#DC143C",    # Cremisi
        "surprise": "#9932CC"  # Viola orchidea
    }
    
    emotion_colors = [colors[e] for e in emotion_values.keys()]
    
    ax.bar(
        emotion_values.keys(),
        emotion_values.values(),
        color=emotion_colors
    )
    
    ax.set_ylim(0, 5.5)
    ax.set_title("Emozioni rilevate dal volto")
    ax.set_yticks(range(6))
    ax.set_yticklabels(["Sconosciuto", "Molto improbabile", "Improbabile", "Possibile", "Probabile", "Molto probabile"])
    
    plt.tight_layout()
    
    return fig

# Funzione principale che viene eseguita quando l'app viene avviata
def main():
    st.title("🎭 Analizzatore di Sentiment e Ironia")
    st.markdown("Analizza il sentiment e rileva l'ironia da testo, immagini e audio utilizzando Google Cloud AI")
    
    # Sidebar per le informazioni
    st.sidebar.header("🔧 Informazioni")
    st.sidebar.markdown("""
    Questo strumento utilizza le API di Google Cloud per analizzare sentiment, ironia ed emozioni.
    
    **Funzionalità disponibili:**
    - Analisi del sentiment e ironia dal testo
    - Rilevamento emozioni da immagini con volti
    - Trascrizione audio e analisi del sentiment
    
    I file predefiniti (test.jpg e test.wav) vengono utilizzati automaticamente se presenti nella directory dell'app.
    """)
    
    # Carica le credenziali dal file .env
    creds_path = load_credentials_from_env()
    
    # Cleanup delle credenziali quando l'app si chiude
    if hasattr(st, 'session_state') and 'creds_cleanup' not in st.session_state:
        st.session_state['creds_cleanup'] = True
        
        def cleanup_temp_creds():
            if creds_path and os.path.exists(creds_path):
                try:
                    os.unlink(creds_path)
                except:
                    pass
        
        # Registro la funzione di cleanup
        import atexit
        atexit.register(cleanup_temp_creds)
    
    # Verifica la presenza dei file predefiniti
    default_files = check_default_files()
    
    # Contenitore principale
    with st.container():
        # Tabs per diverse modalità di analisi
        tabs = st.tabs(["📝 Testo", "📸 Immagine", "🎤 Audio", "🔍 Risultati"])
        
        # Inizializza le variabili per memorizzare i risultati delle analisi
        text_input = ""
        text_results = None
        image_path = None
        image_results = None
        audio_path = None
        audio_results = None
        
        # Tab per l'analisi del testo
        with tabs[0]:
            st.subheader("Analisi del testo")
            text_input = st.text_area(
                "Inserisci il testo da analizzare:",
                value="Fantastico! Sono rimasto bloccato nel traffico per 3 ore, proprio il massimo!",
                height=150
            )
            
            if st.button("Analizza testo", key="analyze_text") and text_input:
                if not creds_path:
                    st.error("❌ File .env non trovato o variabili mancanti. Controlla la configurazione.")
                else:
                    with st.spinner("Analisi del testo in corso..."):
                        text_results = analyze_text_sentiment(text_input)
                        st.session_state['text_results'] = text_results
                        st.session_state['text_input'] = text_input
                        st.success("✅ Analisi del testo completata!")
            
            # Se ci sono risultati, visualizzali
            if 'text_results' in st.session_state:
                st.subheader("Risultati dell'analisi testuale")
                
                # Visualizza il sentiment score con un gauge
                sentiment_fig = plot_sentiment_gauge(st.session_state['text_results']['score'])
                st.pyplot(sentiment_fig)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Intensità dell'emozione (Magnitude)", 
                              f"{st.session_state['text_results']['magnitude']:.2f}")
                with col2:
                    st.metric("Ironia/Sarcasmo rilevato", 
                              "Sì" if st.session_state['text_results']['sarcasm_detected'] else "No")
        
        # Tab per l'analisi dell'immagine
        with tabs[1]:
            st.subheader("Analisi dell'immagine")
            
            # Opzione per utilizzare file predefinito o caricare nuovo file
            use_default_image = False
            if default_files["image"]["exists"]:
                use_default_image = st.checkbox("Usa l'immagine predefinita (test.jpg)", value=True)
                
                if use_default_image:
                    # Mostra l'immagine predefinita
                    default_img_path = default_files["image"]["path"]
                    st.image(default_img_path, caption="Immagine predefinita: test.jpg", use_column_width=True)
                    image_path = default_img_path
                    
                    # Se abbiamo cambiato da un upload a default, cancelliamo l'upload
                    if 'uploaded_image_path' in st.session_state:
                        del st.session_state['uploaded_image_path']
            
            # Se non usiamo il default, mostra l'uploader
            if not use_default_image:
                uploaded_image = st.file_uploader("Carica un'immagine con un volto:", type=["jpg", "jpeg", "png"])
                
                if uploaded_image is not None:
                    # Salva il file caricato
                    image_path = save_uploaded_file(uploaded_image)
                    st.session_state['uploaded_image_path'] = image_path
                    st.image(uploaded_image, caption="Immagine caricata", use_column_width=True)
                elif 'uploaded_image_path' in st.session_state:
                    # Usa l'immagine precedentemente caricata
                    image_path = st.session_state['uploaded_image_path']
                    st.image(image_path, caption="Immagine caricata precedentemente", use_column_width=True)
            
            # Pulsante per analizzare
            if image_path and st.button("Analizza immagine", key="analyze_image"):
                if not creds_path:
                    st.error("❌ File .env non trovato o variabili mancanti. Controlla la configurazione.")
                else:
                    with st.spinner("Analisi dell'immagine in corso..."):
                        image_results = analyze_face_expression(image_path)
                        st.session_state['image_results'] = image_results
                        st.session_state['image_path'] = image_path
                        if "error" not in image_results:
                            st.success("✅ Analisi dell'immagine completata!")
                        else:
                            st.error(f"❌ {image_results['error']}")
            
            # Se ci sono risultati, visualizzali
            if 'image_results' in st.session_state and "error" not in st.session_state['image_results']:
                st.subheader("Risultati dell'analisi dell'immagine")
                
                # Visualizza le emozioni
                emotions_fig = plot_emotions(st.session_state['image_results'])
                if emotions_fig:
                    st.pyplot(emotions_fig)
                
                # Mostra la confidenza del rilevamento
                if "detection_confidence" in st.session_state['image_results']:
                    st.metric("Confidenza del rilevamento", 
                              f"{st.session_state['image_results']['detection_confidence']:.2f}")
        
        # Tab per l'analisi dell'audio
        with tabs[2]:
            st.subheader("Analisi dell'audio")
            
            # Opzione per utilizzare file predefinito o caricare nuovo file
            use_default_audio = False
            if default_files["audio"]["exists"]:
                use_default_audio = st.checkbox("Usa l'audio predefinito (test.wav)", value=True)
                
                if use_default_audio:
                    # Mostra l'audio predefinito
                    default_audio_path = default_files["audio"]["path"]
                    st.audio(default_audio_path)
                    st.info(f"File audio predefinito: {os.path.basename(default_audio_path)}")
                    audio_path = default_audio_path
                    
                    # Se abbiamo cambiato da un upload a default, cancelliamo l'upload
                    if 'uploaded_audio_path' in st.session_state:
                        del st.session_state['uploaded_audio_path']
            
            # Se non usiamo il default, mostra l'uploader
            if not use_default_audio:
                uploaded_audio = st.file_uploader("Carica un file audio:", type=["wav"])
                
                if uploaded_audio is not None:
                    # Salva il file caricato
                    audio_path = save_uploaded_file(uploaded_audio)
                    st.session_state['uploaded_audio_path'] = audio_path
                    st.audio(uploaded_audio)
                elif 'uploaded_audio_path' in st.session_state:
                    # Usa l'audio precedentemente caricato
                    audio_path = st.session_state['uploaded_audio_path']
                    st.audio(audio_path)
            
            # Selezione della lingua
            language_options = {
                "Italiano": "it-IT",
                "Inglese": "en-US",
                "Francese": "fr-FR",
                "Spagnolo": "es-ES",
                "Tedesco": "de-DE"
            }
            selected_language = st.selectbox(
                "Seleziona la lingua dell'audio:",
                options=list(language_options.keys()),
                index=0
            )
            language_code = language_options[selected_language]
            
            # Pulsante per analizzare
            if audio_path and st.button("Analizza audio", key="analyze_audio"):
                if not creds_path:
                    st.error("❌ File .env non trovato o variabili mancanti. Controlla la configurazione.")
                else:
                    with st.spinner("Analisi dell'audio in corso..."):
                        audio_results = transcribe_audio(audio_path, language_code)
                        st.session_state['audio_results'] = audio_results
                        st.session_state['audio_path'] = audio_path
                        if "error" not in audio_results:
                            st.success("✅ Analisi dell'audio completata!")
                        else:
                            st.error(f"❌ {audio_results['error']}")
            
            # Se ci sono risultati, visualizzali
            if 'audio_results' in st.session_state and "error" not in st.session_state['audio_results'] and st.session_state['audio_results'].get('transcript'):
                st.subheader("Risultati dell'analisi dell'audio")
                
                # Mostra la trascrizione
                st.text_area("Trascrizione:", value=st.session_state['audio_results']['transcript'], height=150, disabled=True)
                
                # Mostra la confidenza
                st.metric("Confidenza della trascrizione", 
                          f"{st.session_state['audio_results'].get('confidence', 0):.2f}")
                
                # Se c'è una nota, mostrala
                if "note" in st.session_state['audio_results']:
                    st.info(f"ℹ️ {st.session_state['audio_results']['note']}")
                
                # Analizza il sentiment della trascrizione
                if st.session_state['audio_results']['transcript']:
                    audio_sentiment = analyze_text_sentiment(st.session_state['audio_results']['transcript'])
                    
                    st.subheader("Sentiment della trascrizione")
                    
                    # Visualizza il sentiment score con un gauge
                    sentiment_fig = plot_sentiment_gauge(audio_sentiment['score'])
                    st.pyplot(sentiment_fig)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Intensità dell'emozione (Magnitude)", 
                                  f"{audio_sentiment['magnitude']:.2f}")
                    with col2:
                        st.metric("Ironia/Sarcasmo rilevato", 
                                  "Sì" if audio_sentiment['sarcasm_detected'] else "No")
        
        # Tab per i risultati combinati
        with tabs[3]:
            st.subheader("Analisi Completa")
            
            # Riepilogo dei file selezionati
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📝 Testo: {'Inserito' if 'text_input' in st.session_state else 'Non inserito'}")
            with col2:
                if 'image_path' in st.session_state:
                    img_src = "predefinito (test.jpg)" if os.path.basename(st.session_state['image_path']) == "test.jpg" else "caricato"
                    st.info(f"📸 Immagine: {img_src}")
                else:
                    st.info("📸 Immagine: Non selezionata")
            with col3:
                if 'audio_path' in st.session_state:
                    audio_src = "predefinito (test.wav)" if os.path.basename(st.session_state['audio_path']) == "test.wav" else "caricato"
                    st.info(f"🎤 Audio: {audio_src}")
                else:
                    st.info("🎤 Audio: Non selezionato")
            
            if st.button("Esegui analisi completa", key="analyze_all"):
                if not creds_path:
                    st.error("❌ File .env non trovato o variabili mancanti. Controlla la configurazione.")
                else:
                    with st.spinner("Analisi in corso..."):
                        # Analisi del testo
                        if 'text_input' in st.session_state and st.session_state['text_input']:
                            text_results = analyze_text_sentiment(st.session_state['text_input'])
                            st.session_state['text_results'] = text_results
                        
                        # Analisi dell'immagine
                        if 'image_path' in st.session_state:
                            image_results = analyze_face_expression(st.session_state['image_path'])
                            st.session_state['image_results'] = image_results
                        
                        # Analisi dell'audio
                        if 'audio_path' in st.session_state:
                            audio_results = transcribe_audio(st.session_state['audio_path'], language_code)
                            st.session_state['audio_results'] = audio_results
                        
                        st.success("✅ Analisi completa terminata!")
            
            # Visualizza i risultati in formato tabellare
            if any(k in st.session_state for k in ['text_results', 'image_results', 'audio_results']):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Risultati Testuali")
                    if 'text_results' in st.session_state:
                        results_text = f"""
                        **Testo analizzato:**
                        {st.session_state.get('text_input', 'N/A')}
                        
                        **Sentiment score:** {st.session_state['text_results']['score']:.2f}
                        **Magnitude:** {st.session_state['text_results']['magnitude']:.2f}
                        **Ironia rilevata:** {'✅ Sì' if st.session_state['text_results']['sarcasm_detected'] else '❌ No'}
                        """
                        st.markdown(results_text)
                    else:
                        st.info("Nessuna analisi testuale disponibile")
                
                with col2:
                    st.subheader("📸 Risultati Immagine")
                    if 'image_results' in st.session_state:
                        if "error" not in st.session_state['image_results']:
                            results_image = f"""
                            **Emozioni rilevate:**
                            - 😊 Gioia: {st.session_state['image_results']['joy']}
                            - 😢 Tristezza: {st.session_state['image_results']['sorrow']}
                            - 😠 Rabbia: {st.session_state['image_results']['anger']}
                            - 😲 Sorpresa: {st.session_state['image_results']['surprise']}
                            
                            **Confidenza:** {st.session_state['image_results'].get('detection_confidence', 'N/A')}
                            """
                            st.markdown(results_image)
                        else:
                            st.warning(f"Errore nell'analisi dell'immagine: {st.session_state['image_results']['error']}")
                    else:
                        st.info("Nessuna analisi dell'immagine disponibile")
                
                st.subheader("🎤 Risultati Audio")
                if 'audio_results' in st.session_state:
                    if "error" not in st.session_state['audio_results']:
                        results_audio = f"""
                        **Trascrizione:**
                        {st.session_state['audio_results'].get('transcript', 'Nessuna trascrizione')}
                        
                        **Confidenza:** {st.session_state['audio_results'].get('confidence', 0):.2f}
                        """
                        st.markdown(results_audio)
                        
                        if st.session_state['audio_results'].get('transcript'):
                            audio_sentiment = analyze_text_sentiment(st.session_state['audio_results']['transcript'])
                            results_audio_sentiment = f"""
                            **Sentiment della trascrizione:**
                            - Score: {audio_sentiment['score']:.2f}
                            - Magnitude: {audio_sentiment['magnitude']:.2f}
                            - Ironia rilevata: {'✅ Sì' if audio_sentiment['sarcasm_detected'] else '❌ No'}
                            """
                            st.markdown(results_audio_sentiment)
                    else:
                        st.warning(f"Errore nell'analisi dell'audio: {st.session_state['audio_results']['error']}")
                else:
                    st.info("Nessuna analisi dell'audio disponibile")
    
    # Informazioni aggiuntive nel footer
    st.markdown("---")
    st.markdown("### 📌 Informazioni")
    st.markdown("""
    - Questo strumento utilizza le API di Google Cloud per analizzare sentiment, ironia ed emozioni
    - Le credenziali vengono caricate automaticamente dal file .env nella stessa directory
    - I file predefiniti test.jpg e test.wav vengono utilizzati se presenti nella directory
    - Per i file audio, vengono analizzati al massimo i primi 45 secondi in formato mono
    - Il rilevamento dell'ironia è basato su euristiche e potrebbe non essere sempre accurato
    """)

if __name__ == "__main__":
    main()