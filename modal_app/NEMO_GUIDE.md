# NeMo ASR + VAD Integration Guide

Este documento explica cómo usar los modelos de NVIDIA NeMo con VAD (Voice Activity Detection) integrado en lugar de Whisper + WebRTC VAD.

## Ventajas de NeMo + VAD Integrado

**¿Por qué NeMo en lugar de Whisper?**

1. **VAD Integrado**: Los modelos de NeMo tienen VAD nativo que está optimizado específicamente para trabajar con sus modelos ASR
2. **Mejor Precisión en Español**: Los modelos multilingües de NeMo (especialmente Canary) están específicamente entrenados para múltiples idiomas incluyendo español
3. **Más Rápido**: Los modelos Parakeet son más ligeros y rápidos que Whisper Large
4. **Timestamps Precisos**: El VAD de NeMo produce segmentaciones más precisas basadas en el mismo procesamiento de features que el ASR

## Modelos Disponibles

### Modelos ASR

| Modelo | Tamaño | Velocidad | Idiomas | Uso Recomendado |
|--------|--------|-----------|---------|-----------------|
| `nvidia/parakeet-tdt-0.6b` | 0.6B | Muy rápido | Multilingüe | **Recomendado para español** - Balance perfecto |
| `nvidia/canary-1b` | 1B | Rápido | Multilingüe | Mejor calidad, soporta más idiomas |
| `nvidia/parakeet-tdt-1.1b` | 1.1B | Medio | Multilingüe | Mayor precisión, más recursos |

### Modelos VAD

| Modelo | Descripción |
|--------|-------------|
| `vad_multilingual_marblenet` | **Recomendado** - VAD multilingüe optimizado |
| `vad_multilingual_frame_marblenet` | Variante frame-level, más detalle |

## Configuración Rápida

### 1. Subir Archivos de Audio

```bash
# Subir archivos locales a Modal Volume
modal run modal_app/run_nemo.py::stage_data --audio-folder ./audio
```

### 2. Transcribir un Archivo

```bash
# Transcripción simple con configuración por defecto
modal run modal_app/run_nemo.py::transcribe_single --audio-file ejemplo.webm

# Con modelo Canary para mayor calidad
modal run modal_app/run_nemo.py::transcribe_single \
  --audio-file ejemplo.webm \
  --model nvidia/canary-1b \
  --language es
```

### 3. Transcripción por Lotes

```bash
# Transcribir todos los archivos con configuración por defecto
modal run modal_app/run_nemo.py::batch_transcription

# Con configuración personalizada
modal run modal_app/run_nemo.py::batch_transcription \
  --model nvidia/parakeet-tdt-0.6b \
  --language es \
  --vad-onset 0.7 \
  --vad-offset 0.5
```

## Configuración del VAD

El VAD de NeMo tiene varios parámetros que afectan cómo se detecta el habla:

### Parámetros Principales

| Parámetro | Rango | Por Defecto | Descripción |
|-----------|-------|-------------|-------------|
| `--vad-onset` | 0.0-1.0 | 0.5 | Umbral para **iniciar** un segmento de habla. Mayor = más conservador |
| `--vad-offset` | 0.0-1.0 | 0.5 | Umbral para **terminar** un segmento de habla. Mayor = segmentos más cortos |
| `--vad-min-duration-on` | segundos | 0.1 | Duración mínima de habla para considerar un segmento |
| `--vad-min-duration-off` | segundos | 0.3 | Duración mínima de silencio para separar segmentos |

### Presets Recomendados

#### Conservador (Mayor Precisión)

Usa esto cuando quieras menos segmentos pero de mayor calidad:

```bash
modal run modal_app/run_nemo.py::batch_transcription \
  --vad-onset 0.7 \
  --vad-offset 0.7 \
  --vad-min-duration-on 0.3 \
  --vad-min-duration-off 0.5
```

**Resultado**: Menos segmentos, ignora ruidos breves, bueno para audio con mucho ruido de fondo.

#### Balanceado (Por Defecto)

Configuración equilibrada para la mayoría de casos:

```bash
modal run modal_app/run_nemo.py::batch_transcription \
  --vad-onset 0.5 \
  --vad-offset 0.5 \
  --vad-min-duration-on 0.1 \
  --vad-min-duration-off 0.3
```

**Resultado**: Balance entre capturar todo el habla y evitar ruido.

#### Sensible (Máxima Captura)

Usa esto para capturar incluso pausas y titubeos:

```bash
modal run modal_app/run_nemo.py::batch_transcription \
  --vad-onset 0.3 \
  --vad-offset 0.3 \
  --vad-min-duration-on 0.05 \
  --vad-min-duration-off 0.2
```

**Resultado**: Más segmentos, captura todo incluyendo respiraciones y pausas.

## Ejemplo Completo: Pipeline de Transcripción

```bash
# 1. Subir archivos
modal run modal_app/run_nemo.py::stage_data --audio-folder ./audio

# 2. Ver archivos disponibles
modal run modal_app/run_nemo.py::list_files

# 3. Probar con un archivo
modal run modal_app/run_nemo.py::transcribe_single \
  --audio-file entrevista_01.webm \
  --model nvidia/parakeet-tdt-0.6b \
  --language es \
  --vad-onset 0.6

# 4. Si funciona bien, procesar todos
modal run modal_app/run_nemo.py::batch_transcription \
  --model nvidia/parakeet-tdt-0.6b \
  --language es \
  --vad-onset 0.6 \
  --vad-offset 0.5

# 5. Descargar resultados
modal volume get transcription-results /results ./resultados
```

## Formato de Salida

Los resultados se guardan en formato JSON con esta estructura:

```json
{
  "audio_path": "entrevista_01.webm",
  "language": "es",
  "segments": [
    {
      "start": 0.5,
      "end": 5.2,
      "text": "Hola, bienvenidos a esta entrevista sobre inteligencia artificial.",
      "words": null
    },
    {
      "start": 6.1,
      "end": 12.8,
      "text": "Hoy hablaremos sobre los avances en procesamiento del lenguaje natural.",
      "words": null
    }
  ]
}
```

## Integración con Base de Datos

Para guardar los resultados en la base de datos de `fast_audio_annotate`:

```python
import json
from pathlib import Path
from db_backend import get_db, insert_clip

# Leer resultado de transcripción
with open("resultados/entrevista_01_transcription.json") as f:
    transcription = json.load(f)

# Conectar a DB
db = get_db()

# Insertar cada segmento como un clip
for segment in transcription["segments"]:
    insert_clip(
        db=db,
        audio_path=transcription["audio_path"],
        start_timestamp=segment["start"],
        end_timestamp=segment["end"],
        text=segment["text"],
        username="nemo-asr",
        marked=False
    )

print(f"Insertados {len(transcription['segments'])} clips en la base de datos")
```

## Comparación: NeMo vs Whisper

| Característica | NeMo + VAD | Whisper + WebRTC |
|----------------|------------|------------------|
| **VAD** | Integrado, mismo modelo | Externo (WebRTC) |
| **Velocidad (Parakeet)** | 🚀🚀🚀 Muy rápido | 🚀 Rápido |
| **Precisión en Español** | 🎯🎯🎯 Excelente | 🎯🎯 Muy buena |
| **Segmentación** | Más precisa | Buena |
| **Configuración VAD** | Parámetros ML | Parámetros heurísticos |
| **Modelos disponibles** | 3 (Parakeet, Canary) | Muchos (Whisper family) |
| **Word timestamps** | ❌ No soportado | ✅ Sí |

## Troubleshooting

### Error: "No speech detected"

**Causa**: El VAD es demasiado conservador.

**Solución**: Baja los umbrales:
```bash
--vad-onset 0.3 --vad-offset 0.3
```

### Error: "Too many segments"

**Causa**: El VAD es demasiado sensible.

**Solución**: Aumenta los umbrales:
```bash
--vad-onset 0.7 --vad-offset 0.7 --vad-min-duration-off 0.5
```

### Audio con mucho ruido de fondo

**Solución**: Usa configuración conservadora:
```bash
--vad-onset 0.8 --vad-min-duration-on 0.3
```

### Quiero capturar pausas y titubeos

**Solución**: Usa configuración sensible:
```bash
--vad-onset 0.2 --vad-min-duration-on 0.05
```

## Próximos Pasos

1. **Ajusta parámetros VAD**: Experimenta con diferentes umbrales según tu tipo de audio
2. **Prueba diferentes modelos**: Compara Parakeet vs Canary para tu caso de uso
3. **Integra con tu flujo de trabajo**: Usa los JSON de salida en tu aplicación

## Referencias

- [NeMo Toolkit Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [NeMo ASR Models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/collections/nemo_asr)
- [Frame VAD Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/vad.html)
