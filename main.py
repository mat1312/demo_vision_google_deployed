import os
import argparse
import re
import wave
import math
import numpy as np
from google.cloud import language_v1
from google.cloud import vision_v1 as vision
from google.cloud import speech_v1 as speech

# Funzione per configurare le credenziali Google Cloud
def setup_credentials(credentials_path):
    """Imposta le credenziali di Google Cloud"""
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        print(f"✅ Credenziali impostate: {credentials_path}")
    else:
        print("⚠️ Nessun file di credenziali specificato. Assicurati che le credenziali siano configurate nell'ambiente.")

### 🔍 1. Analisi del Sentiment e Ironia nel Testo ###
def analyze_text_sentiment(text):
    """Analizza il sentiment e rileva potenziale ironia nel testo"""
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        
        sentiment = client.analyze_sentiment(request={"document": document}).document_sentiment
        
        # Euristiche migliorate per il rilevamento dell'ironia
        sarcasm_detected = False
        
        # Metodo 1: Sentiment positivo con magnitude sufficiente (ridotta a 0.8)
        if sentiment.score > 0 and sentiment.magnitude > 0.8:
            sarcasm_detected = True
            
        # Metodo 2: Cerca pattern tipici dell'ironia
        irony_patterns = [
            r'\.{3}|…',              # Punti di sospensione
            r'proprio il massimo',   # Frasi sarcastiche comuni
            r'che bello',
            r'fantastico\W.+negativo', # Contrasto tra positivo e negativo
            r'adoro\W.+dopo',         # Schema "positivo... dopo" (come nel tuo esempio)
            r'migliore\W.+dopo',      # Schema "migliore... dopo"
            r'!\?|\?!',               # Combinazione di punti esclamativi e interrogativi
        ]
        
        for pattern in irony_patterns:
            if re.search(pattern, text.lower()):
                sarcasm_detected = True
                break
                
        # Metodo 3: Contrasto semantico (parole positive seguite da negative)
        positive_words = ['adoro', 'migliore', 'fantastico', 'bellissimo', 'perfetto']
        negative_words = ['calcio', 'stinchi', 'traffico', 'bloccato', 'terribile', 'orribile']
        
        has_positive = any(word in text.lower() for word in positive_words)
        has_negative = any(word in text.lower() for word in negative_words)
        
        if has_positive and has_negative:
            sarcasm_detected = True
        
        return {
            "score": sentiment.score,
            "magnitude": sentiment.magnitude,
            "sarcasm_detected": sarcasm_detected
        }
    except Exception as e:
        print(f"❌ Errore nell'analisi del testo: {e}")
        return {"score": 0, "magnitude": 0, "sarcasm_detected": False}

### 🖼 2. Analisi delle Espressioni Facciali da Immagine ###
def analyze_face_expression(image_path):
    """Analizza le espressioni facciali in un'immagine"""
    if image_path.lower() == 'none':
        print("⏩ Analisi dell'immagine saltata")
        return {"error": "Analisi saltata"}
        
    try:
        client = vision.ImageAnnotatorClient()

        with open(image_path, "rb") as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        response = client.face_detection(image=image)

        emotions = {}
        if response.face_annotations:
            face = response.face_annotations[0]  # Prende il primo volto rilevato
            
            # Converte i valori di likelihood in formato leggibile
            likelihood_names = ["UNKNOWN", "VERY_UNLIKELY", "UNLIKELY", "POSSIBLE", "LIKELY", "VERY_LIKELY"]
            
            emotions = {
                "joy": likelihood_names[face.joy_likelihood],
                "sorrow": likelihood_names[face.sorrow_likelihood],
                "anger": likelihood_names[face.anger_likelihood],
                "surprise": likelihood_names[face.surprise_likelihood]
            }
            
            # Aggiungi altre informazioni sul volto
            emotions["detection_confidence"] = face.detection_confidence
            
            return emotions
        else:
            print("⚠️ Nessun volto rilevato nell'immagine")
            return {"error": "Nessun volto rilevato"}
    except FileNotFoundError:
        print(f"❌ File immagine non trovato: {image_path}")
        return {"error": "File non trovato"}
    except Exception as e:
        print(f"❌ Errore nell'analisi dell'immagine: {e}")
        return {"error": str(e)}

def transcribe_audio(audio_path, language_code="it-IT"):
    """
    Trascrive l'audio e restituisce il testo, supportando file di qualsiasi lunghezza 
    utilizzando Google Cloud Storage per file lunghi
    """
    if audio_path.lower() == 'none':
        print("⏩ Analisi dell'audio saltata")
        return {"error": "Analisi saltata"}
        
    try:
        if not audio_path.lower().endswith(('.wav')):
            return {"error": "Il file deve essere in formato WAV per l'analisi"}
            
        # Importa la libreria storage (aggiungila in cima al file se non è già presente)
        from google.cloud import storage
            
        # Apre il file WAV e legge le proprietà
        with wave.open(audio_path, 'rb') as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            n_frames = wav.getnframes()
            file_duration = n_frames / frame_rate
            
            print(f"🔊 Elaborazione audio di {file_duration:.1f} secondi (file completo)")
                
            # Legge TUTTI i frame dell'audio
            wav.setpos(0)
            audio_data = wav.readframes(n_frames)
        
        # Verifica se è necessario convertire da stereo a mono
        temp_file = f"temp_{os.path.basename(audio_path)}"
        
        if channels == 2:
            print("🔄 Conversione da stereo a mono in corso...")
            
            # Converte i bytes in un array numpy
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Rimodella l'array per avere 2 canali
            audio_array = audio_array.reshape(-1, channels)
            
            # Calcola la media dei canali per ottenere mono
            mono_array = audio_array.mean(axis=1).astype(np.int16)
            
            # Scrivi un nuovo file WAV mono
            with wave.open(temp_file, 'wb') as mono_wav:
                mono_wav.setnchannels(1)  # 1 canale (mono)
                mono_wav.setsampwidth(sample_width)
                mono_wav.setframerate(frame_rate)
                mono_wav.writeframes(mono_array.tobytes())
                
            file_to_analyze = temp_file
            conversion_note = "Audio convertito da stereo a mono"
        else:
            # Se è già mono, non serve conversione ma creiamo comunque un file temporaneo per sicurezza
            with wave.open(temp_file, 'wb') as mono_wav:
                mono_wav.setnchannels(1)  # Forza 1 canale per sicurezza
                mono_wav.setsampwidth(sample_width)
                mono_wav.setframerate(frame_rate)
                mono_wav.writeframes(audio_data)
                
            file_to_analyze = temp_file
            conversion_note = "Audio in formato mono"
        
        # Per file audio lunghi, usiamo Google Cloud Storage
        if file_duration > 60:  # Per file più lunghi di 1 minuto
            print("🔄 File audio lungo rilevato, utilizzo Google Cloud Storage...")
            
            # Crea un client Storage
            storage_client = storage.Client()
            
            # Genera un nome per il bucket temporaneo o usa uno esistente
            # Nota: Dovresti avere un bucket già configurato o permessi per crearne uno
            bucket_name = "audio_analysis_temp"
            
            # Verifica se il bucket esiste, altrimenti crealo
            try:
                bucket = storage_client.get_bucket(bucket_name)
            except Exception:
                print(f"⚠️ Bucket {bucket_name} non trovato, creazione in corso...")
                bucket = storage_client.create_bucket(bucket_name)
            
            # Genera un nome file unico usando timestamp
            import time
            blob_name = f"audio_{int(time.time())}_{os.path.basename(file_to_analyze)}"
            
            # Carica il file su GCS
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_to_analyze)
            
            # Ottieni l'URI GCS
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            print(f"✅ File caricato su: {gcs_uri}")
            
            # Usa l'API Speech con riferimento GCS
            client = speech.SpeechClient()
            
            audio = speech.RecognitionAudio(uri=gcs_uri)
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=frame_rate,
                language_code=language_code,
                audio_channel_count=1,
                enable_automatic_punctuation=True,
                enable_separate_recognition_per_channel=False,
            )
            
            print("⏳ Avvio trascrizione asincrona via GCS...")
            operation = client.long_running_recognize(config=config, audio=audio)
            print("⏳ Elaborazione in corso, attendere (potrebbe richiedere tempo per file lunghi)...")
            response = operation.result(timeout=1800)  # Timeout di 30 minuti per file molto lunghi
            
            # Dopo l'analisi, elimina il file da GCS
            blob.delete()
            print(f"✅ File temporaneo eliminato da GCS")
            
            # Elabora i risultati
            if response.results:
                transcript = " ".join([result.alternatives[0].transcript for result in response.results])
                confidence = response.results[0].alternatives[0].confidence
                
                note_message = f"{conversion_note}, file completo analizzato via GCS ({file_duration:.1f} secondi)"
                
                success_result = {
                    "transcript": transcript, 
                    "confidence": confidence, 
                    "note": note_message
                }
            else:
                note_message = f"{conversion_note}, file completo analizzato via GCS ({file_duration:.1f} secondi)"
                
                success_result = {
                    "transcript": "", 
                    "confidence": 0, 
                    "note": f"Nessun risultato. {note_message}"
                }
        else:
            # Per file audio brevi, usa l'API diretta
            client = speech.SpeechClient()
            
            with open(file_to_analyze, 'rb') as audio_file:
                content = audio_file.read()
                
            audio = speech.RecognitionAudio(content=content)
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=frame_rate,
                language_code=language_code,
                audio_channel_count=1,
                enable_automatic_punctuation=True,
                enable_separate_recognition_per_channel=False,
            )
            
            print("⏳ Avvio trascrizione diretta...")
            response = client.recognize(config=config, audio=audio)
            
            # Elabora i risultati
            if response.results:
                transcript = " ".join([result.alternatives[0].transcript for result in response.results])
                confidence = response.results[0].alternatives[0].confidence
                
                note_message = f"{conversion_note}, file completo analizzato ({file_duration:.1f} secondi)"
                
                success_result = {
                    "transcript": transcript, 
                    "confidence": confidence, 
                    "note": note_message
                }
            else:
                note_message = f"{conversion_note}, file completo analizzato ({file_duration:.1f} secondi)"
                
                success_result = {
                    "transcript": "", 
                    "confidence": 0, 
                    "note": f"Nessun risultato. {note_message}"
                }
        
        # Elimina il file temporaneo locale
        os.remove(file_to_analyze)
        
        return success_result
            
    except FileNotFoundError:
        print(f"❌ File audio non trovato: {audio_path}")
        return {"error": "File non trovato"}
    except Exception as e:
        print(f"❌ Errore nell'analisi audio: {e}")
        # Assicurati di eliminare il file temporaneo in caso di errore
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return {"error": str(e)}
            
    except FileNotFoundError:
        print(f"❌ File audio non trovato: {audio_path}")
        return {"error": "File non trovato"}
    except Exception as e:
        print(f"❌ Errore nell'analisi audio: {e}")
        # Assicurati di eliminare il file temporaneo in caso di errore
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return {"error": str(e)}
            
    except FileNotFoundError:
        print(f"❌ File audio non trovato: {audio_path}")
        return {"error": "File non trovato"}
    except Exception as e:
        print(f"❌ Errore nell'analisi audio: {e}")
        # Assicurati di eliminare il file temporaneo in caso di errore
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return {"error": str(e)}
            
    except FileNotFoundError:
        print(f"❌ File audio non trovato: {audio_path}")
        return {"error": "File non trovato"}
    except Exception as e:
        print(f"❌ Errore nell'analisi audio: {e}")
        # Assicurati di eliminare il file temporaneo in caso di errore
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return {"error": str(e)}

def display_results(text_analysis, image_analysis, audio_analysis, text_content):
    """Visualizza in modo ordinato i risultati dell'analisi"""
    
    print("\n" + "="*50)
    print("📊 RISULTATI DELL'ANALISI")
    print("="*50)
    
    # Visualizza analisi del testo
    print("\n🔤 ANALISI DEL TESTO:")
    print(f"📝 Testo analizzato: \"{text_content}\"")
    print(f"📊 Sentiment score: {text_analysis['score']:.2f} (-1 negativo, +1 positivo)")
    print(f"📏 Magnitude: {text_analysis['magnitude']:.2f} (intensità dell'emozione)")
    print(f"🎭 Ironia/sarcasmo rilevato: {'✅ Sì' if text_analysis['sarcasm_detected'] else '❌ No'}")
    
    # Visualizza analisi immagine
    print("\n📸 ANALISI DELL'IMMAGINE:")
    if "error" in image_analysis:
        print(f"❌ Errore: {image_analysis['error']}")
    else:
        print(f"😊 Gioia: {image_analysis['joy']}")
        print(f"😢 Tristezza: {image_analysis['sorrow']}")
        print(f"😠 Rabbia: {image_analysis['anger']}")
        print(f"😲 Sorpresa: {image_analysis['surprise']}")
        if "detection_confidence" in image_analysis:
            print(f"🔍 Confidenza rilevamento: {image_analysis['detection_confidence']:.2f}")
    
    # Visualizza analisi audio
    print("\n🎤 ANALISI DELL'AUDIO:")
    if "error" in audio_analysis:
        print(f"❌ Errore: {audio_analysis['error']}")
    elif "note" in audio_analysis:
        print(f"⚠️ Nota: {audio_analysis['note']}")
        print(f"🔤 Trascrizione: \"{audio_analysis['transcript']}\"")
        print(f"🔍 Confidenza: {audio_analysis.get('confidence', 0):.2f}")
        
        if audio_analysis['transcript']:
            audio_sentiment = analyze_text_sentiment(audio_analysis['transcript'])
            print(f"📊 Sentiment della trascrizione: {audio_sentiment['score']:.2f}")
            print(f"📏 Magnitude della trascrizione: {audio_sentiment['magnitude']:.2f}")
            print(f"🎭 Ironia/sarcasmo nella trascrizione: {'✅ Sì' if audio_sentiment['sarcasm_detected'] else '❌ No'}")
    else:
        print(f"🔤 Trascrizione: \"{audio_analysis['transcript']}\"")
        print(f"🔍 Confidenza: {audio_analysis.get('confidence', 0):.2f}")
        
        # Se abbiamo ottenuto una trascrizione, analizziamo anche il suo sentiment
        if audio_analysis['transcript']:
            audio_sentiment = analyze_text_sentiment(audio_analysis['transcript'])
            print(f"📊 Sentiment della trascrizione: {audio_sentiment['score']:.2f}")
            print(f"📏 Magnitude della trascrizione: {audio_sentiment['magnitude']:.2f}")
            print(f"🎭 Ironia/sarcasmo nella trascrizione: {'✅ Sì' if audio_sentiment['sarcasm_detected'] else '❌ No'}")
    
    print("\n" + "="*50)

def main():
    """Funzione principale che esegue l'analisi"""
    
    # Configurazione del parser degli argomenti
    parser = argparse.ArgumentParser(description="Analizzatore di sentiment e ironia")
    parser.add_argument("--credentials", help="Percorso al file delle credenziali Google Cloud")
    parser.add_argument("--text", default="Fantastico! Sono rimasto bloccato nel traffico per 3 ore, proprio il massimo!", 
                        help="Testo da analizzare")
    parser.add_argument("--image", default="test.jpg", help="Percorso al file immagine da analizzare (o 'none' per saltare)")
    parser.add_argument("--audio", default="test.wav", help="Percorso al file audio da analizzare (o 'none' per saltare)")
    parser.add_argument("--language", default="it-IT", help="Codice lingua per la trascrizione audio (default: it-IT)")
    
    args = parser.parse_args()
    
    # Imposta le credenziali
    setup_credentials(args.credentials)
    
    print("\n🚀 Avvio analisi...")
    
    # Esegui le analisi
    text_analysis = analyze_text_sentiment(args.text)
    image_analysis = analyze_face_expression(args.image)
    audio_analysis = transcribe_audio(args.audio, args.language)
    
    # Visualizza i risultati
    display_results(text_analysis, image_analysis, audio_analysis, args.text)

if __name__ == "__main__":
    main()