## 1. Introduction

### Problem Description
Il **Progetto 6: Valutazione e Fine-Tuning di Modelli Linguistici (LLM)** ha come obiettivo principale l'analisi comparativa di modelli linguistici pre-addestrati, valutandone le capacità di adattamento a task specifici. Il focus del lavoro risiede nell'esecuzione pratica di diverse strategie di **Fine-Tuning** su modelli distinti, al fine di misurarne quantitativamente le prestazioni post-adattamento e farne emergere, tramite confronto diretto, i rispettivi **punti di forza e di debolezza**.

Nello specifico, il nostro gruppo ha selezionato il task di **Emotion Recognition** (Classificazione delle Emozioni) applicato a testi brevi provenienti dai social media. La rilevanza di questo problema è elevata nell'attuale panorama digitale: la capacità di analizzare automaticamente il "sentiment" e le sfumature emotive (gioia, tristezza, rabbia, ecc.) su piattaforme come Twitter è cruciale per applicazioni di monitoraggio della brand reputation, analisi sociale e customer care automatizzato.
Il pubblico di riferimento include ricercatori NLP, data scientist e sviluppatori di assistenti virtuali empatici. Una soluzione efficace permette di automatizzare l'analisi di grandi moli di dati non strutturati (Big Data) con costi ridotti rispetto all'annotazione manuale.

### Proposed Solution
Per affrontare il problema, abbiamo adottato un **approccio sperimentale comparativo**, testando diverse tecniche di fine-tuning su una selezione eterogenea di modelli pre-addestrati. L'obiettivo centrale è stato confrontare l'efficacia delle diverse strategie di adattamento in relazione alle architetture sottostanti.

La nostra soluzione si articola nei seguenti punti:

* **Modelli Selezionati:** Abbiamo condotto esperimenti su quattro architetture distinte per valutare l'impatto della dimensione e della specializzazione del pre-training:
    * `vinai/bertweet-base`: Modello domain-specific per Twitter.
    * `distilbert/distilbert-base-uncased`: Modello distillato (leggero) per testare l'efficienza.
    * `google-bert/bert-base-uncased`: La baseline standard per il NLP.
    * `FacebookAI/roberta-base`: Versione ottimizzata di BERT.

* **Approccio Metodologico:** Un confronto sistematico tra tre metodologie di apprendimento:
    1.  **In-Context Learning (Zero/Few-Shot):** Per testare le capacità native senza aggiornamento dei pesi.
    2.  **Transfer Learning (Frozen):** Addestramento del solo classificatore mantenendo il corpo del modello congelato.
    3.  **LoRA (Low-Rank Adaptation):** Tecnica PEFT che inietta matrici addestrabili nei layer del modello.

* **Sfide Computazionali:** Le principali difficoltà affrontate sono state la gestione della **Context Window** limitata (tipica dei modelli Encoder-Only), che ha imposto vincoli severi sulle strategie di Few-Shot Learning, e la sfida di massimizzare le performance di generalizzazione nonostante il **basso numero di parametri** dei modelli "base" e "distilled" selezionati, richiedendo un'ottimizzazione fine degli iperparametri.

**Task Distribution:**
Il lavoro è stato parallelizzato assegnando a ciascun membro del gruppo la responsabilità completa di una specifica architettura. Ogni componente ha curato l'analisi esplorativa, il pre-processing, il fine-tuning e la valutazione del proprio modello, per poi convergere in una fase finale di confronto dei risultati.

* **[Membro 1]:** Responsabile per `vinai/bertweet-base`.
* **[Membro 2]:** Responsabile per `distilbert/distilbert-base-uncased`.
* **[Membro 3]:** Responsabile per `google-bert/bert-base-uncased`.
* **[Membro 4]:** Responsabile per `FacebookAI/roberta-base`.

**Summary of Achieved Results:**
TODO

### `[extra]` Literature Review
In letteratura, il benchmark sul dataset `dair-ai/emotion` è dominato da approcci di Full Fine-Tuning basati su modelli massivi come RoBERTa-Large o BERTweet (es. il modello `Emanuel/bertweet-emotion-base`), che raggiungono accuracy attorno al 94.5%. Tuttavia, questi approcci richiedono risorse computazionali significative e sono soggetti a *Catastrophic Forgetting*. Il nostro lavoro si inserisce nel filone di ricerca sull'efficienza (PEFT), dimostrando come tecniche come LoRA, se ben calibrate (targettando tutti i layer lineari), possano colmare il gap prestazionale con i metodi tradizionali a una frazione del costo.

## 2. Proposed Method

### Solution Choice
Abbiamo scartato l'uso di LLM generativi generalisti (es. GPT-4, Llama-3) per motivi di latenza in inferenza e costi computazionali, preferendo un'architettura **Encoder-Only** (`BERTweet`). I modelli Encoder sono strutturalmente superiori nei task di classificazione pura grazie all'attenzione bidirezionale che cattura l'intero contesto della frase simultaneamente.

Tra le tecniche di addestramento, la scelta è ricaduta su **LoRA (Low-Rank Adaptation)** come soluzione primaria. A differenza del *Linear Probing* (che congela il corpo del modello limitandone l'apprendimento) e del *Full Fine-Tuning* (che è lento e pesante), LoRA permette di adattare le rappresentazioni interne del modello al task specifico delle emozioni modificando solo poche matrici di peso aggiuntive. Questo garantisce modularità e portabilità.

### Methodology for performance measurement
Le prestazioni sono state valutate utilizzando un set di metriche standard per la classificazione multiclasse:
* **Accuracy:** Per una visione globale dell'efficacia del modello.
* **Macro F1-Score:** Metrica critica scelta a causa dello sbilanciamento del dataset (le classi *Joy* e *Sadness* sono molto più frequenti di *Surprise* e *Love*). L'F1-Score garantisce che il modello non stia semplicemente ignorando le classi minoritarie.
* **Confusion Matrix:** Utilizzata per l'analisi qualitativa degli errori, per identificare quali coppie di emozioni vengono confuse più spesso (es. *Love* vs *Joy*).

## 3. Experimental Results

### Demonstration and Technologies
Per garantire la riproducibilità degli esperimenti, il progetto è stato sviluppato utilizzando le seguenti tecnologie e versioni:
* **Environment:** Google Colab (Python 3.10, GPU T4 16GB VRAM).
* **Librerie:**
    * `transformers` (HuggingFace) per la gestione del modello BERTweet.
    * `peft` (HuggingFace) per l'implementazione di LoRA.
    * `datasets` e `evaluate` per la gestione dei dati e delle metriche.
    * `scikit-learn` per il calcolo delle matrici di confusione.
* **Istruzioni Demo:** Il notebook fornito esegue automaticamente il download del dataset, il caricamento dei pesi LoRA ottimizzati e la valutazione sul validation set, stampando a video le metriche finali.

### Results
La configurazione migliore identificata (**LoRA Optimized**) ha ottenuto i seguenti risultati sul Validation Set:
* **Accuracy:** 93.35%
* **Macro F1-Score:** 88.30%
* **Training Time:** ~15 minuti (10 epoche).

Questa configurazione utilizzava un Rango $r=8$, un Learning Rate di $1e^{-3}$ con *Cosine Scheduler*, e applicava LoRA a **tutti i layer lineari** del trasformatore (`query`, `key`, `value`, `dense`), non solo a quelli di attenzione.

### Ablation Study: Comparison across configurations
Lo studio di ablation è stato fondamentale per isolare il contributo delle diverse componenti. Di seguito il confronto tra le configurazioni testate:

| Configurazione | Accuracy | F1-Score | Analisi dell'Impatto |
| :--- | :--- | :--- | :--- |
| **In-Context (k=0)** | 49.30% | 24.6% | Il modello comprende la lingua ma fallisce nelle sfumature emotive specifiche (F1 basso). |
| **In-Context (k=5)** | 48.88% | 25.7% | **Performance degradate**. A causa del limite di 128 token, l'aggiunta di esempi ha causato il troncamento del prompt in 25 casi su 200. |
| **Transfer Learning** | 54.45% | 29.9% | (Frozen Body). Insufficiente. Le feature interne di BERTweet non sono linearmente separabili per le 6 emozioni senza adattamento profondo. |
| **LoRA Base** | 91.15% | 86.8% | (Target: solo Q, V). Ottimo salto prestazionale rispetto al frozen. |
| **LoRA Optimized** | **93.35%** | **88.3%** | (Target: All Linear). Estendere l'adattamento ai layer `dense` e `key` è stato determinante per raggiungere lo Stato dell'Arte. |

### `[extra]` Comparative Study with Literature
Confrontando il nostro modello *LoRA Optimized* (93.35%) con il modello di riferimento della letteratura `Emanuel/bertweet-emotion-base` (Full Fine-Tuning, ~94.5%), osserviamo un gap prestazionale minimo (<1.2%). Tuttavia, il nostro approccio ha richiesto l'addestramento di soli **1.8 Milioni** di parametri contro i **135 Milioni** del modello full, dimostrando una netta superiorità in termini di efficienza parametrica.

## 4. Discussion and Conclusions

### Results Discussion
I risultati ottenuti hanno superato le aspettative iniziali per un approccio PEFT, confermando che i modelli domain-specific come BERTweet sono basi eccellenti per il transfer learning. L'analisi qualitativa degli errori (Matrice di Confusione) ha rivelato che la maggior parte degli errori residui si concentra su due coppie:
1.  **Love vs Joy:** Dovuto alla forte sovrapposizione semantica e all'uso di parole simili.
2.  **Surprise vs Fear:** Entrambe emozioni ad alta attivazione (arousal), spesso confuse in contesti ambigui.

### Method Validity
Il metodo LoRA si è dimostrato estremamente valido e robusto. Al contrario, lo studio di ablation su **In-Context Learning** ha evidenziato un limite strutturale critico: l'architettura BERTweet, progettata con una finestra di contesto ridotta (128 token), **non è idonea** a strategie Few-Shot ($k>3$). Gli esempi dimostrativi tendono a saturare la memoria e troncare il prompt, peggiorando le performance invece di migliorarle, rendendo il fine-tuning una scelta obbligata per questo tipo di architetture.

### Limitations and Maturity
* **Limiti:** La *Context Window* limitata impedisce l'uso di prompt complessi o catene di ragionamento (CoT). Il dataset presenta un sbilanciamento naturale che penalizza leggermente le classi minoritarie (*Surprise*).
* **Maturità:** La soluzione sviluppata, basata su librerie standard industriali (`transformers`, `peft`), ha raggiunto un livello di maturità tecnologica (TRL) elevato, pronta per essere integrata in pipeline di produzione con costi di inferenza contenuti.

### Future Works
Per avanzare ulteriormente nel progetto, proponiamo:
1.  **Data Augmentation mirata:** Utilizzare tecniche di Back-Translation (Inglese -> Francese -> Inglese) specificamente per le classi *Surprise* e *Love* per ridurre lo sbilanciamento e la confusione semantica.
2.  **Supervised Contrastive Learning:** Introdurre una fase di pre-training con loss contrastiva per forzare una maggiore separazione vettoriale tra le classi simili prima del fine-tuning.
3.  **Ensemble Learning:** Combinare le predizioni probabilistiche di BERTweet con un modello RoBERTa-Large per sfruttare i punti di forza complementari dei due modelli.