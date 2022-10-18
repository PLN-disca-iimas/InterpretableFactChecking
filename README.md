# :newspaper: Automatic Fact Checking Using an Interpretable Bert-Based Architecture on COVID-19 Claims
![GitHub](https://img.shields.io/github/license/PLN-disca-iimas/InterpretableFactChecking)
![GitHub repo size](https://img.shields.io/github/repo-size/PLN-disca-iimas/InterpretableFactChecking)
![GitHub last commit](https://img.shields.io/github/last-commit/PLN-disca-iimas/InterpretableFactChecking)
![GitHub stars](https://img.shields.io/github/stars/PLN-disca-iimas/InterpretableFactChecking)

This repository contains the implementation of a neural network architecture focused on verifying facts against evidence found in a knowledge base. The architecture can perform relevance evaluation and claim verification. We fine-tuned BERT to codify claims and pieces of evidence separately. An attention layer between the claim and evidence representation computes alignment scores to identify relevant terms between both. Finally, a classification layer receives the vector representation of claims and evidence and performs the relevance and verification classification. Our model allows a more straightforward interpretation of the predictions than other state-of-the-art models. We use the scores computed within the attention layer to show which evidence spans are more relevant to classify a claim as supported or refuted. We use the model to verify facts about COVID-19. The COVID-19 facts corpus is also provided [here](https://github.com/PLN-disca-iimas/InterpretableFactChecking/tree/main/dataset).

### :pencil: How to cite


### :neckbeard: Collaborators
Ramón Casillas (PCIC-UNAM), Helena Gómez-Adorno (IIMAS-UNAM), Victor Lomas-Barrie (IIMAS-UNAM) and Orlando Ramos-Flores (IIMAS-UNAM)

### :heavy_check_mark: Aknowledgments
