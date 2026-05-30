# noisy-retina-project

Projeto experimental, simples e modular, para testar aprendizado com rotulos ruidosos em imagens medicas usando o Kaggle Diabetic Retinopathy original.

O treino real foi pensado para rodar no Kaggle com GPU. Localmente, o foco e rodar smoke tests leves para validar imports, codigo e pipeline.

## Estrutura

```text
noisy-retina-project/
  configs/
    smoke.yaml
    smoke_sce.yaml
    smoke_qmix_like.yaml
    kaggle_small_ce.yaml
    kaggle_small_sce.yaml
    kaggle_small_qmix_like.yaml
  src/
    adapters/
      __init__.py
      lnmbench_adapter.py
    datasets/
      __init__.py
      kaggle_dr.py
    models/
      __init__.py
      resnet.py
    methods/
      __init__.py
      ce.py
      sce.py
      qmix_like.py
    utils/
      __init__.py
      seed.py
      metrics.py
      checkpoint.py
    __init__.py
    train.py
  notebooks/
    kaggle_launcher.py
  README.md
  requirements.txt
```

## Instalar

```bash
conda create --prefix ./venv python=3.12.4
conda activate ./venv
```

```bash
cd noisy-retina-project
pip install -r requirements.txt
```

## Validar imports

```bash
python -c "import src.train; import src.datasets.kaggle_dr; import src.models.resnet; import src.methods.ce; import src.methods.sce; import src.methods.qmix_like; import src.adapters.lnmbench_adapter"
```

## Smoke test local

O smoke test usa `data/smoke/trainLabels.csv`, um CSV fake pequeno, e cria imagens RGB falsas quando os arquivos reais nao existem.

Para validar somente dataset e DataLoader, sem treinar modelo:

```bash
python src/train.py --config configs/smoke.yaml --dry-run
```

Saida esperada:

```text
train_batches=2 val_batches=1
batch_image_shape=(4, 3, 224, 224)
batch_labels=tensor([4, 0, 2, 0])
```

Treino minimo com CE:

```bash
python src/train.py --config configs/smoke.yaml
```

Treino minimo com SCE:

```bash
python src/train.py --config configs/smoke_sce.yaml
```

Treino experimental QMix-like:

```bash
python src/train.py --config configs/smoke_qmix_like.yaml
```

Os outputs serao salvos em:

```text
outputs/smoke_ce/metrics.json
outputs/smoke_ce/last_checkpoint.pt
outputs/smoke_sce/metrics.json
outputs/smoke_sce/last_checkpoint.pt
outputs/smoke_qmix_like/metrics.json
outputs/smoke_qmix_like/last_checkpoint.pt
outputs/smoke_qmix_like/group_report.json
```

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

## QMix-like

`qmix_like` e um metodo inspirado em separacao por confiabilidade, nao uma implementacao completa do QMix original. Ele treina com CE no warm-up, calcula um score por amostra (`CE + entropia`), separa as amostras em grupos `clean`, `medium` e `noisy`, e usa pesos 1.0, 0.5 e 0.2 nas epocas seguintes.

## LNMBench

`src/adapters/lnmbench_adapter.py` mantem uma interface minima para integracao futura com LNMBench sem copiar, importar ou modificar o codigo interno do LNMBench.
