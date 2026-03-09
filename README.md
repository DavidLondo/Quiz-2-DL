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

### Decisiones de diseño del pipeline

Adaptar el esquema del notebook de referencia (VGG16 + frutas) al dataset de
Sign Language MNIST requirió tres ajustes técnicos no triviales:

- **Grayscale → pseudo-RGB:** ResNet18 espera tensores de 3 canales. Las imágenes
  del dataset son monocromáticas (1 canal), por lo que se replica el canal único
  tres veces (`img.repeat(3, 1, 1)`). Esto no añade información nueva, pero
  satisface la interfaz de la arquitectura sin alterar los pesos preentrenados.

- **Resolución 64×64 en lugar de 224×224:** ResNet fue entrenado con imágenes de
  224×224. Sin embargo, las imágenes originales son de 28×28 píxeles — escalar
  directamente a 224×224 introduce artefactos severos de interpolación sobre
  imágenes que originalmente tienen muy poca resolución. Se eligió 64×64 como
  compromiso: suficiente para que la red extraiga features espaciales sin
  distorsión extrema, y manejable en CPU sin cambiar la lógica del pipeline.

- **Remap de etiquetas:** Las etiquetas originales van de 0 a 24 saltando el 9
  (letra J, excluida por requerir movimiento). PyTorch espera índices contiguos
  de 0 a N-1 para `CrossEntropyLoss`. Se construye un mapa `{0:0, ..., 8:8,
  10:9, ..., 24:23}` que convierte las 24 etiquetas no contiguas al rango
  continuo `[0, 23]`.

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

**Baseline CNN:** El modelo muestra 79.33% de train accuracy y 99.07% de val accuracy, lo cual
parece contradictorio: normalmente se espera que el modelo rinda mejor en train
que en val. La explicación está en el data augmentation: las transformaciones
de flip horizontal y rotación aleatoria se aplican **únicamente al conjunto de
entrenamiento**. Esto hace que cada imagen de train que ve el modelo durante
el forward pass sea una versión alterada y más difícil de la original, mientras
que la validación recibe imágenes limpias sin transformar. El modelo aprende a
generalizar bien precisamente porque entrenó con versiones difíciles, pero su
accuracy de train medido sobre esas mismas versiones difíciles es artificialmente
bajo. No es overfitting inverso — es el efecto esperado de un augmentation bien
aplicado.

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

¿Por qué ResNet18 y no VGG16?

El notebook de referencia usa VGG16 (~138 M parámetros). Se eligió ResNet18
(~11 M parámetros) por dos razones técnicas:

1. **Eficiencia computacional:** una diferencia de 12× en parámetros es
   significativa al entrenar en CPU. ResNet18 permite completar los experimentos
   en tiempos razonables sin sacrificar la capacidad de representación necesaria
   para este dataset.

2. **Estabilidad del fine-tuning:** ResNet usa conexiones residuales (skip
   connections) que permiten que el gradiente fluya directamente desde capas
   profundas hacia capas tempranas sin degradarse. Esto hace que descongelar
   `layer4` durante el fine-tuning sea más estable que hacerlo en VGG16, donde
   los gradientes deben atravesar secuencias largas de capas sin atajos y son
   más propensos a desvanecerse o explotar.

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

## El gap val→test como indicador de generalización real

La comparación de val accuracy entre modelos puede ser engañosa. Exp1 alcanza
96.47% de val accuracy — un número que podría parecer satisfactorio — pero su
test accuracy cae a 79.32%, un gap de **−17.15 puntos porcentuales**. Exp2
tiene 99.80% de val accuracy y 96.57% de test accuracy, un gap de solo
**−3.23 puntos**.

| Modelo | Val Acc | Test Acc | Gap val→test |
|--------|---------|----------|-------------|
| Baseline CNN | 99.07 % | 91.01 % | −8.06 pp |
| Exp1 — solo fc | 96.47 % | 79.32 % | **−17.15 pp** |
| Exp2 — fine-tuning | 99.80 % | 96.57 % | −3.23 pp |

Este gap mide qué tan frágil es la generalización del modelo. En Exp1, el
backbone congelado aprende a mapear features de ImageNet hacia las 24 clases
de signos, pero ese mapeo depende de que los inputs de train y val provengan
de la misma distribución interna del dataset. Cuando el modelo se evalúa
sobre el conjunto de test — una partición diferente con posibles variaciones
de iluminación, posición de mano o contraste — el mapeo aprendido sobre
features inadecuadas se rompe con mayor facilidad.

En Exp2, `layer4` ajusta sus filtros para capturar características realmente
relevantes para el dominio (bordes de dedos, orientaciones de mano, contornos
de gestos). Esas representaciones son más robustas ante variaciones de
distribución entre splits, lo que se refleja en un gap pequeño y estable.

**Conclusión práctica:** al comparar modelos de Transfer Learning, el gap
val→test es más informativo que la val accuracy aislada. Un gap grande indica
que el modelo está explotando patrones específicos de la partición de validación
pero no ha aprendido representaciones genuinamente transferibles al problema.

---

## Mejor modelo recomendado

**Experimento 2 — Transfer Learning con Fine-Tuning** es la estrategia óptima
para este problema, logrando **96.57%** de test accuracy con 14 épocas totales
(8 warm-up + 6 fine-tuning), superando tanto al Baseline CNN (91%) como al
Transfer Learning sin fine-tuning (79%).
