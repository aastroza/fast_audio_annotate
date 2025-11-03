# Guía de Transcripción con Modal

Este proyecto usa Modal para ejecutar transcripciones de Whisper en la nube con segmentación automática usando VAD (Voice Activity Detection).

## Características

- ✅ **Transcripción en Modal**: Usa GPUs L40S en la nube (configurable)
- ✅ **Segmentación VAD**: Divide automáticamente el audio en segmentos con timestamps
- ✅ **Word Timestamps**: Opcionalmente obtiene timestamps a nivel de palabra
- ✅ **Batching Automático**: Procesa múltiples segmentos en paralelo
- ✅ **Caché de Modelos**: Los modelos se cachean en Modal Volume para cargas rápidas

## Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2. Configura Modal:
```bash
modal setup
```

3. (Opcional) Configura tu token de Hugging Face si usas modelos privados:
```bash
export HF_TOKEN=tu_token_aqui
```

## Uso Básico

### 1. Transcribir un directorio de audio

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --model openai/whisper-large-v3 \
  --language es \
  --batch-size 8
```

Esto:
- Transcribe todos los archivos de audio en `./audio`
- Guarda los resultados JSON en `./audio/transcriptions/`
- Almacena los segmentos en la base de datos SQLite (`./audio/annotations.db`)

### 2. Con word timestamps

Para obtener timestamps a nivel de palabra:

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --model openai/whisper-large-v3 \
  --language es \
  --word-timestamps
```

### 3. Sin segmentación VAD

Para transcribir cada archivo como un solo bloque (sin VAD):

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --model openai/whisper-large-v3 \
  --language es \
  --no-vad
```

### 4. Sobrescribir transcripciones existentes

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --overwrite
```

## Opciones de Configuración

### Argumentos de línea de comandos

- `--config`: Ruta al archivo de configuración YAML (default: `./config.yaml`)
- `--audio-folder`: Carpeta con archivos de audio
- `--output`: Directorio para guardar JSONs de transcripción (default: `{audio-folder}/transcriptions`)
- `--model`: Modelo de Whisper a usar (default: del config.yaml)
- `--language`: Idioma de transcripción (ej: `es`, `en`, `auto`)
- `--batch-size`: Tamaño de batch para inferencia (default: 8)
- `--word-timestamps`: Incluir timestamps a nivel de palabra
- `--no-vad`: Desactivar segmentación VAD
- `--overwrite`: Sobrescribir transcripciones existentes
- `--database-url`: URL de base de datos (para usar Postgres en lugar de SQLite)
- `--modal`: Forzar uso de Modal (default)
- `--no-modal`: Ejecutar Whisper localmente en lugar de Modal

## Formato de Salida

### JSON de transcripción

Cada archivo de audio genera un JSON con esta estructura:

```json
{
  "audio_path": "path/to/audio.wav",
  "relative_audio_path": "audio.wav",
  "model": "openai/whisper-large-v3",
  "language": "es",
  "duration": 120.5,
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Hola, esto es una prueba.",
      "words": [
        {
          "start": 0.0,
          "end": 0.5,
          "text": "Hola"
        },
        {
          "start": 0.6,
          "end": 1.0,
          "text": "esto"
        }
      ]
    }
  ]
}
```

### Base de datos

Los segmentos también se guardan en la base de datos con estos campos:

- `audio_path`: Ruta relativa del audio
- `start_timestamp`: Inicio del segmento (segundos)
- `end_timestamp`: Fin del segmento (segundos)
- `text`: Transcripción del segmento
- `username`: Usuario que creó la transcripción
- `timestamp`: Timestamp de creación
- `marked`: Flag booleano para marcar clips

## Configuración Avanzada

### Personalizar parámetros VAD

Puedes ajustar la segmentación VAD editando los parámetros en `src/fast_audio_annotate/transcription.py`:

```python
DEFAULT_SERVER_VAD = {
    "aggressiveness": 2,        # 0-3, más alto = más agresivo
    "frame_ms": 30,             # Tamaño de frame en ms
    "min_speech_ms": 150,       # Mínimo de voz para iniciar segmento
    "min_silence_ms": 300,      # Mínimo de silencio para terminar segmento
    "max_chunk_ms": 30000,      # Máximo tamaño de chunk (30s)
    "padding_ms": 200,          # Padding al inicio/fin de cada segmento
}
```

### Cambiar GPU en Modal

Edita `src/fast_audio_annotate/modal_transcription.py`:

```python
@app.cls(gpu="L40S", timeout=60*10, scaledown_window=5, max_containers=10)
```

Opciones de GPU:
- `"L4"`: GPU económica, 24GB VRAM
- `"L40S"`: Recomendada, 48GB VRAM (default)
- `"A100"`: GPU potente, 40GB/80GB VRAM
- `"H100"`: GPU más potente, 80GB VRAM

### Usar base de datos PostgreSQL

Para producción, puedes usar Neon o cualquier PostgreSQL:

```bash
export NEON_DATABASE_URL="postgresql://user:pass@host/db"

python scripts/preprocess_audio.py \
  --audio-folder ./audio
```

O especificar directamente:

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --database-url "postgresql://user:pass@host/db"
```

## Troubleshooting

### Error: "No module named 'modal'"

```bash
pip install modal==1.2.1
```

### Error: Modal authentication

```bash
modal setup
# Luego sigue las instrucciones para autenticarte
```

### Los segmentos son muy largos

Ajusta `max_chunk_ms` en `DEFAULT_SERVER_VAD` a un valor menor (ej: 15000 para 15s)

### Los segmentos son muy cortos

Aumenta `min_speech_ms` para requerir más tiempo de voz antes de crear un segmento

### Modelo no encontrado

Asegúrate de que el modelo existe en Hugging Face:
- `openai/whisper-large-v3`
- `openai/whisper-large-v3-turbo`
- `openai/whisper-medium`
- `openai/whisper-small`

## Arquitectura

```
┌─────────────────────────────────────────────┐
│  Local: preprocess_audio.py                │
│  - Carga audio                              │
│  - Aplica VAD para crear segmentos         │
│  - Envía segmentos a Modal                 │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Modal: WhisperModel                        │
│  - GPU: L40S                                │
│  - Batched inference (hasta 64 samples)     │
│  - Retorna transcripciones con timestamps   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Local: Guardar resultados                  │
│  - JSON con segmentos                       │
│  - Base de datos (SQLite/PostgreSQL)        │
└─────────────────────────────────────────────┘
```

## Comparación: Local vs Modal

| Aspecto              | Local (--no-modal) | Modal (default)     |
|----------------------|--------------------|---------------------|
| GPU requerida        | ✅ Sí              | ❌ No               |
| Velocidad            | Depende del HW     | Rápido (L40S)       |
| Costo                | $0 (tu HW)         | ~$1.20/hora GPU     |
| Paralelización       | Limitada           | ✅ Auto-scaling     |
| Setup                | CUDA, drivers, etc | `modal setup`       |

## Ejemplos de Uso

### Ejemplo 1: Transcripción básica en español

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./podcasts \
  --model openai/whisper-large-v3 \
  --language es
```

### Ejemplo 2: Análisis detallado con word timestamps

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./interviews \
  --model openai/whisper-large-v3-turbo \
  --language es \
  --word-timestamps \
  --batch-size 16
```

### Ejemplo 3: Re-procesar con modelo diferente

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --model openai/whisper-medium \
  --overwrite
```

## Contribuir

Para reportar issues o contribuir, visita el repositorio en GitHub.

## Licencia

Ver archivo LICENSE.
