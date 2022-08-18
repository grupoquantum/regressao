from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.regression import Regression
from Neuraline.Utilities.chart import Chart
regression, chart = Regression(), Chart()

inputs = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
outputs = [[9], [22], [27], [44], [45], [66], [63], [88], [81], [110]]
regression.fit(
    inputs=inputs,
    outputs=outputs, 
    degree=None, # número real correspondente ao grau de inclinação da linha de regressão com base na extremidade final.
    alpha=None, # número real correspondente ao nível de horizontalidade da linha de regressão, a horizontalidade máxima corresponde a 0.5.
    same_output=False, # se definido como True irá retornar a saída do treinamento que mais se aproxima do resultado da predição.
    only_integers=False, # se definido como True irá aproximar os resultados para convertê-los em números inteiros.
    count=False, # se definido como True aplicará o cálculo de regressão para dados de contagem retornando apenas números inteiros maiores ou iguais a zero.
    nonlinear=True, # se definido como True além de resultados lineares retornará também resultados não lineares.
    outliers=True, # se definido como True permanecerá com os dados originais, se definido como False irá remover os outliers (valores discrepantes).
)
test_inputs = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
test_outputs = [[9], [22], [27], [44], [45], [66], [63], [88], [81], [110]]
predicted_outputs = regression.predict(inputs=test_inputs)
print(predicted_outputs)

chart.plotMATRIX(matrix1=test_outputs, matrix2=predicted_outputs)