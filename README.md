# K-Driver Slicing

Repositório para minha tentativa de ajudar o @phmouras a reescrever o código de
seu projeto de mestrado.

Na forma atual, o código está altamente dependente de variáveis efetivamente
globais, além de muita coisa não fazer muito sentido estruturalmente. Isso se
deve ao código original estar em notebook (ipynb). Marquei os trechos de código
correspondentes pela numeração de blocos (ex.: `# BLOCK 1`).

Em geral, os nomes de funções e variáveis que eu criei não seguem nenhuma
convenção específica e podem não ser precisos para o que fazem.

Para tentar reduzir a verbosidade, importei explicitamente as funções
utilizadas, especialmente do `numpy`. Até onde pude verificar, não há conflitos
de namespacing.

Algumas decisões (uso de `copy()`, por exemplo) foram baseadas na intenção de
melhorar a performance, mas não verifiquei com benchmarking se fazem sentido.
Seria interessante verificar, especialmente por em geral a `numpy` ser uma
biblioteca muito bem otimizada.

No geral, não mexi nos trechos de visualização, que usam `matplotlib`. Preciso
ler melhor como a API funciona para entender se faz sentido reescrever alguns
trechos, especialmente pelas visualizações terem muitas labels e granularidades
de dados diferentes. Mas olhando superficialmente, vários trechos parecem ser
simplificáveis.

Parece haver algumas variáveis não utilizadas ou replicadas desnecessariamente;
é necessário olhar com mais cuidado. Possivelmente, as variáveis paramétricas
poderiam ser melhor organizadas também.

Lembrando que essas observações foram feitas no contexto de rodar o código
reescrito como um programa Python normal; nem tudo se aplica ao código em
`ipynb`, que provavelmente é mais interessante para uma apresentação. De
qualquer modo, uma estrutura mais encapsulada e robusta pode reduzir o ruído de
informação no arquivo.

## Notas

**GPU**:
Apesar do `numpy` já aplicar multithreading, como há muitas operações em
matrizes que nem sempre dependem umas das outras linearmente, pode ser
interessante executá-las na GPU. A biblioteca [CuPy](https://cupy.dev) poderia,
talvez, auxiliar nesse sentido sem grandes esforços, mas é necessário olhar com
calma.

**Corrigir a função `chebyshev()`:**
Apesar de, até onde eu consigo ver, a minha implementação ser análoga ao código
original, minha versão gera erro se N != M, onde o código original funciona sem
problemas aparentes.
