Obiettivi:
- Sviluppare un modello non supervisionato in grado di identificare e localizzare i difetti nei tappeti dalle immagini.
- Valutare le performance sulla sezione carpet del dataset MVTec AD misurando l'AUROC.

Tecniche:
- Architettura: Convolutional Autoencoders (CAE) e/o Variational Autoencoders (VAE).
- Training: Addestramento effettuato esclusivamente su immagini defect-free.
- Anomaly Scoring: Calcolo dell'errore di ricostruzione (es. tramite MSE o SSIM) per la generazione delle anomaly maps.
