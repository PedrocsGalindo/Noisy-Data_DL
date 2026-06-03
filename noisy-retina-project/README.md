# noisy-retina-project

## Dependencias 

```bash
conda create --prefix ./venv python=3.12.4
conda activate ./venv
```
```bash
cd noisy-retina-project
pip install -r requirements.txt
```

*Base line repo*

``` bash
git clone https://github.com/myyy777/LNMBench.git
cd LNMBench
pip install -r requirements.txt
```
no windows pode dar errado, usar esse comando:
```bash
git clone --no-checkout https://github.com/myyy777/LNMBench.git LNMBench
cd LNMBench
git sparse-checkout init --no-cone
git sparse-checkout set "/*" "!/**/results/**"
git checkout
```

*Dataset*
> https://www.kaggle.com/c/diabetic-retinopathy-detection

Nao eh o kaggle DR+, eh o kaggle DR, o + vem de outro trabalho feito onde contraram mais medicos para rotular. sendo uma base reanotada/reengenheirada.  

## Justificativa:
> Mesmo domínio retinal do QMix, ruído real como no LNMBench, mas sem evidência de teste direto do QMix nesse dataset.
    
O próprio repositório oficial do QMix lista DRTiD DeepDRiD, EyeQ, DRAC e ODIR como datasets utilzados

### Rodar 
Escolha o arquivo de configuracao de acordo com o experimento ou teste que deseja realizar
>python src/train.py --config configs/arquivo.yaml

- Para testar com poucos dados:    
    
    Precisa baixar as imagens no link do datset, o subset com nome de sample.zip e o trainLabels.csv.zip. E colocar todos os arquivos de imagem em data/sample/images e as labels no mesmo data/sample

- Smoke test local
  
  O smoke test usa `data/smoke/trainLabels.csv`, um CSV fake pequeno, e cria imagens RGB falsas quando os arquivos reais nao existem.

## Kaggle

Antes de rodar no Kaggle, ajuste nos YAMLs os campos:

```yaml
dataset:
  csv_path: /kaggle/input/diabetic-retinopathy-detection/trainLabels.csv
  image_dir: /kaggle/input/diabetic-retinopathy-detection/train
```

Comandos por metodo:

```bash
python src/train.py --config configs/kaggle_small_ce.yaml
python src/train.py --config configs/kaggle_small_sce.yaml
python src/train.py --config configs/kaggle_small_qmix_like.yaml
```

Tambem e possivel usar o launcher:

```bash
python notebooks/kaggle_launcher.py --method ce
python notebooks/kaggle_launcher.py --method sce
python notebooks/kaggle_launcher.py --method qmix_like
```

## QMix
https://ieeexplore.ieee.org/abstract/document/10971353


## LNMBench