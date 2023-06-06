# GBN Font Classificator

Projeto criado com o objetivo de treinar um modelo capaz de classificar as fontes disponiveis no GBN Dataset.

# Estrutura do projeto
### Architectures
As arquiteturas de redes neurais utilizadas para o treinamento.

### Dataset
Dividido em duas partes, a primeira "base_dataset" contem o dataset original e suas divisões posteriores (AFT e Other). A segunda parte "divided_dataset" contém os datasets da primeira parte divididos em "Train", "Validation" e "Test".

### Enums
Enums criados com as informações de diretório e alguns hyparameters que são utilizados no treinamento de todas as arquiteturas de rede.

### Logs
Atualmente usado apenas para gravar as classes que estão sendo treinadas.

### Models
Onde são salvos os modelos após o treinamento, seguindo a estrutura "models/{nome da arquitetura}/{nome do modelo}"

O nome do modelo atualmente segue o padrão "model_id_{id no modelo na planilha de treinamento}.h5"

### Utils
Códigos que podem vir a ser uteis no treinamento das redes.

# Como treinar?

Antes de tudo, é necessário ativar o ambiente conda:
```
conda activate gbnv1.15
```

Com o ambiente ativo, basta instaciar a classe da arquitetura desejada, e preencher o construtor da mesma com as informações contidas no enum. Para rodar utilize:
```
python3 main.py
```

# TODO
[] Transformar a arquitetura RESNET em uma classe
[] Criar os testes indicados pelo Lucas