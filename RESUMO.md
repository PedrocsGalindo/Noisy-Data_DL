## Dataset - medico
### Datasets citados no artigo

|Dataset|Tipo no artigo|Modalidade / contexto|
|---|---|---|
|**PathMNIST**|Balanceado|Histopatologia / colorectal cancer histology|
|**DermaMNIST**|Desbalanceado|Dermatoscopia / lesões de pele|
|**BloodMNIST**|Desbalanceado|Microscopia / células sanguíneas|
|**OrganCMNIST**|Desbalanceado|CT / órgãos em fatias coronais|
|**DRTiD**|Ruído real|Fundus images / retinopatia diabética|
|**Kaggle DR+**|Ruído real|Fundus images / retinopatia diabética|
|**CheXpert**|Ruído real|Raio-X de tórax|
### Métodos principais avaliados

|Categoria|Métodos|
|---|---|
|**Baseline**|**CE / Cross-Entropy**|
|**Noise transition matrix estimation**|**T-Revision**, **VolMinNet**|
|**Noise-robust regularization**|**SCE**, **CDR**|
|**Sample selection**|**Co-teaching**, **Co-teaching+**, **CoDis**, **JoCoR**|
|**Semi-supervised**|**DISC**, **DivideMix**|
# Projeto 
Usar o QMix em um dataset que nao eh reportado como utilizado no repositório oficial (Kaggle DR)

- **CE**: treina normalmente, confiando nos rótulos.
- **SCE**: ainda treina normalmente, mas troca a loss para ser mais robusta a rótulos errados. (**Symmetric Cross Entropy**)
- **QMix-like**: não é só uma loss; é uma estratégia de treinamento que tenta separar amostras limpas, ruidosas e de baixa qualidade.

#### QMix
1. label noise  -> o rótulo está errado  
2. data/image quality noise  -> a imagem é ruim, ambígua ou não tem informação visual suficiente
> Ele alterna entre duas etapas: separação de amostras e treinamento semi-supervisionado quality-aware. A separação usa loss e incerteza para separar amostras corretamente rotuladas, amostras mal rotuladas com boa qualidade e amostras mal rotuladas com baixa qualidade.