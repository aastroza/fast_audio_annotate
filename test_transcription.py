#!/usr/bin/env python3
"""
Script de prueba para verificar que la transcripción con Modal funcione correctamente.
"""
from pathlib import Path
import sys

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.modal_transcription import ModalWhisperTranscriber


def test_transcription():
    """Test básico de transcripción con Modal."""

    print("="*80)
    print("TEST DE TRANSCRIPCIÓN CON MODAL")
    print("="*80)

    # Configuración
    model_name = "openai/whisper-large-v3"
    language = "es"

    print(f"\nConfiguración:")
    print(f"  Modelo: {model_name}")
    print(f"  Idioma: {language}")
    print(f"  VAD: Activado (auto)")
    print(f"  Word timestamps: Desactivado")

    # Crear transcriber
    print("\nInicializando transcriber con Modal...")
    transcriber = ModalWhisperTranscriber(
        model_name=model_name,
        language=language,
        return_word_timestamps=False,
        chunking_strategy="auto",  # Usa VAD automáticamente
        batch_size=8,
    )
    print("✓ Transcriber inicializado")

    # Buscar un archivo de audio para probar
    audio_dir = Path("./audio")
    if not audio_dir.exists():
        print(f"\n❌ Error: No existe el directorio {audio_dir}")
        print("   Crea un directorio 'audio' y coloca archivos .wav o .mp3 para probar")
        return

    # Encontrar primer archivo de audio
    audio_files = list(audio_dir.glob("**/*.wav")) + list(audio_dir.glob("**/*.mp3"))
    if not audio_files:
        print(f"\n❌ Error: No se encontraron archivos de audio en {audio_dir}")
        print("   Agrega archivos .wav o .mp3 al directorio 'audio'")
        return

    test_file = audio_files[0]
    print(f"\nArchivo de prueba: {test_file}")

    # Transcribir
    print("\nTranscribiendo (esto puede tomar unos minutos en la primera ejecución)...")
    print("  - Modal descargará el modelo si no está en caché")
    print("  - La GPU se iniciará automáticamente")
    print("  - El audio se segmentará usando VAD")

    try:
        result = transcriber.transcribe_file(test_file)

        print("\n" + "="*80)
        print("RESULTADOS")
        print("="*80)

        print(f"\nIdioma detectado: {result.language or 'N/A'}")
        print(f"Número de segmentos: {len(result.segments)}")

        print("\nSegmentos:")
        for i, segment in enumerate(result.segments, 1):
            duration = segment.end - segment.start
            print(f"\n  [{i}] {segment.start:.2f}s - {segment.end:.2f}s (duración: {duration:.2f}s)")
            print(f"      \"{segment.text}\"")

            if segment.words:
                print(f"      Palabras: {len(segment.words)}")

        # Texto completo
        full_text = " ".join(s.text for s in result.segments)
        print(f"\nTexto completo:\n  {full_text}")

        print("\n" + "="*80)
        print("✓ Prueba completada exitosamente")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error durante la transcripción:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nPara procesar múltiples archivos, usa:")
    print("  python scripts/preprocess_audio.py --audio-folder ./audio")


if __name__ == "__main__":
    test_transcription()
