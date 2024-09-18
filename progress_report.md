# Relatório de progresso

Aqui é feita a documentação das etapas do desafio.

## Quebra da imagem em blocos

Primeiramente a imagem foi quebrada em blocos de 256x256 para geração de dataset. Foi utilizado o opencv para isso e o formato de jpg. As que ficaram com largura ou altura menor que 160 foram descartadas e foi realizado um redimensionamento das restantes para que todas tivessem esse tamanho.

## Geração do dataset
Etapas de geração do dataset/

### Geração da máscara binária

Para treinar um modelo de segmentação é necessária uma máscara binária. Ela foi gerada utilizando clusterização dos pixels em 3 grupos. Mais próximos de verde, mais próximos de marrom e indefinidos. Depois foi feito um pós processamento de dilatação e erosão.

Foi feita uma tentiva usando o GLI primeiro, mas o resultado não foi muito bom.

O resultado pode ser visto nas imagens a seguir.

![](pictures/comparison_1.png)
![](pictures/comparison_2.png)

### Dataset de treino e validação

Foram separadas 10% das imagens para validação.

### Aumento do dataset

Foram utilizadas técnicas de aumento como rotação, flip, etc.

### Implementação da rede neural

### Treinamento da rede neural

32 épocas com batch de 32 o resultado provavelmente overfitou.

Mudança no prós processamento das máscaras ajudou a melhorar generalização.

