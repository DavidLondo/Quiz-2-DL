# Transfer Learning — Sign Language MNIST
### SI-3014 · Análisis Comparativo de Experimentos

---

## Dataset

| Campo | Valor |
|-------|-------|
| Dataset | Sign Language MNIST (Laboratorio 1) |
| Carpeta | `archive/` |
| Train | 27 455 muestras |
| Test | 7 172 muestras |
| Clases | 24 letras (A–Y, sin J ni Z) |
| Formato | CSV — imágenes 28×28 grayscale |

---

## Modelos evaluados

| # | Modelo | Arquitectura | Paráms entrenables | Épocas |
|---|--------|-------------|-------------------|--------|
| 0 | **Baseline CNN** | CNN propia (3 ConvBlocks) | ~2.2 M (100%) | 10 |
| 1 | **Exp1 — TL solo última capa** | ResNet18 congelado + fc nuevo | 12 312 (0.11%) | 8 |
| 2 | **Exp2 — TL Fine-Tuning** | ResNet18 layer4+fc descongelados | 8 406 040 (75.1%) | 8+6=14 |

---

## 1. Accuracies de entrenamiento y validación

| Modelo | Train Acc (última época) | Val Acc (última época) | Test Acc |
|--------|--------------------------|------------------------|----------|
| Baseline CNN | 79.33 % | 99.07 % | **91.01 %** |
| Exp1 — Solo última capa | 95.56 % | 96.47 % | **79.32 %** |
| Exp2 — Fine-Tuning (FaseB) | 99.64 % | 99.80 % | **96.57 %** |

**Baseline CNN:** Existe un gap pronunciado entre train (79%) y val (99%).
Esto se explica porque el train usa data augmentation (flip + rotación), lo que hace
las imágenes de entrenamiento más difíciles. El modelo generaliza bien en validación
pero cae a 91% en test, evidenciando cierta diferencia de distribución entre splits.

**Experimento 1:** Train y val convergen de forma muy estable y cercana (~95–96%),
sin señales de overfitting. La brecha train/val es la más pequeña de los tres modelos.
Sin embargo, el test accuracy cae drásticamente a 79.32%, revelando que el backbone
congelado de ResNet18 (entrenado en fotos naturales de ImageNet) no extrae
representaciones suficientemente útiles para gestos de manos en escala de grises.

**Experimento 2:** Presenta la convergencia más rápida y las métricas más altas.
En la Fase B (fine-tuning) el modelo pasa de ~96% a ~99.8% de val accuracy en la
primera época, una mejora drástica. El gap train/val es mínimo (99.64% vs 99.80%),
lo que indica que el modelo generaliza correctamente. El test accuracy de 96.57% es
el mejor de los tres.

---

## 2. Transfer Learning vs. modelo previamente desarrollado (Baseline CNN)

| Aspecto | Baseline CNN | Exp1 (TL sin FT) | Exp2 (TL con FT) |
|---------|-------------|------------------|------------------|
| Conocimiento inicial | Pesos aleatorios | Features ImageNet | Features ImageNet |
| Velocidad de convergencia | Lenta | Rápida (88% en época 1) | Muy rápida (99%+ en FaseB época 1) |
| Riesgo de overfitting | Moderado | Muy bajo | Bajo |
| Adaptación al dominio | Total | Ninguna | Alta (layer4 entrenado) |
| Test Accuracy | 91.01 % | 79.32 % ❌ | 96.57 % ✅ |

El Transfer Learning **sin** fine-tuning (Exp1) fue **peor** que la CNN propia
(79% vs 91% en test). El dominio de Sign Language MNIST es muy diferente al de
ImageNet: imágenes en escala de grises, 28×28 píxeles, centradas en gestos de
manos. El backbone congelado extrae features de fotos naturales que no capturan
la especificidad de este dataset.

El Transfer Learning **con** fine-tuning (Exp2) superó claramente al Baseline
(97% vs 91% en test). Al permitir que `layer4` se adapte al nuevo dominio,
el modelo combina el conocimiento general de ImageNet con representaciones
específicas al problema, obteniendo el mejor resultado general.

**Conclusión:** El valor del Transfer Learning no es universal; depende de la
similitud entre el dominio de origen y el dominio destino. Cuando los dominios
difieren, el fine-tuning es imprescindible.

---

## 3. Efecto de entrenar solo la última capa vs. entrenar más capas

### Entrenar solo la última capa (Exp1 — backbone congelado)

- **Parámetros entrenables:** 12 312 (apenas el 0.11% del modelo)
- **Qué aprende:** únicamente una proyección lineal desde los features fijos de ImageNet hacia las 24 clases.
- **Ventajas:**
  - Muy rápido de entrenar (~35 s/época en CPU)
  - Sin riesgo de "olvidar" el preentrenamiento (catastrophic forgetting)
  - Converge sin overfitting
- **Desventajas:**
  - Las representaciones internas nunca se adaptan al dominio destino
  - Test accuracy limitado al 79.32%
- **Cuándo funciona bien:** cuando el dominio destino es similar al de origen (ej. clasificar otro tipo de fotos naturales con ImageNet como base).

### Entrenar más capas — Fine-Tuning (Exp2 — layer4 + fc)

- **Parámetros entrenables:** 8 406 040 (75.1% del modelo) con lr=1e-4
- **Qué aprende:** la capa convolucional final (`layer4`) reajusta sus filtros para detectar patrones relevantes en gestos de manos, mientras preserva los aprendizajes de capas más tempranas.
- **Ventajas:**
  - Test accuracy de 96.57%, el más alto
  - Convergencia extremadamente rápida en Fase B (1 época para pasar de 96% a 99.8% val)
  - Gap train/val mínimo → buena generalización
- **Desventajas:**
  - Más costoso computacionalmente (~2 min/época en CPU en Fase B)
  - Requiere una Fase A de warm-up previa para estabilizar la cabeza antes de descongelar capas profundas; sin esto, el lr alto destruiría los pesos preentrenados
- **Cuándo es necesario:** cuando el dominio destino difiere significativamente del origen, como en este caso.

### Comparación directa

| | Exp1 (solo fc) | Exp2 (layer4+fc) |
|---|---|---|
| Paráms entrenables | 12 K | 8.4 M |
| Tiempo/época | ~35 s | ~35 s (FaseA) / ~130 s (FaseB) |
| Val Acc final | 96.47 % | **99.80 %** |
| Test Acc | 79.32 % | **96.57 %** |
| Gap val→test | −17.15 pp | −3.23 pp |

El gap val→test es el indicador más importante: 17 puntos en Exp1 vs solo 3 puntos
en Exp2, lo que confirma que el fine-tuning produce un modelo que generaliza
genuinamente al dominio del problema, mientras que el backbone congelado
"memoriza" un mapeo frágil desde features inapropiadas.

---

## Mejor modelo recomendado

**Experimento 2 — Transfer Learning con Fine-Tuning** es la estrategia óptima
para este problema, logrando **96.57%** de test accuracy con 14 épocas totales
(8 warm-up + 6 fine-tuning), superando tanto al Baseline CNN (91%) como al
Transfer Learning sin fine-tuning (79%).