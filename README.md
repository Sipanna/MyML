Во время изучения алгоритмов машинного обучения, чтобы лучше разобраться во всех методах и запомнить их, я решила их самостоятельно реализовать.
Здесь представлена моя реализация  алгоритмов.

<h1>linear_models</h1>
<ul>
  <li><b>LinearRegression</b> - аналитическое решение линейной регрессии<br></li>
  <li><b>RidgeRegression</b> - аналитическое решение для линейной регрессии с L2-регуляризацией<br></li>
  <li><b>GDRegression</b> - линейная регрессия через градиентный спуск<br></li>
  <li><b>SGDRegression</b> - стохастический градиентный спуск<br>
  &nbsp;&nbsp;&nbsp;&nbsp;В классе реализованы методы:<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fit - каждый шаг по случайному признаку<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fit_batch - каждый шаг по минибатчам<br></li>
  
</ul>

<h1>svm</h1>
<ul>
<li><b>LinearSVM</b> - линейный метод опорных векторов. Функция потерь минимизируется методом градиентного спуска.<br></li>
<li><b>KernelSVM_SSMO</b> - ядровой метод опорных векторов. Используется упрощенный алгоритм SMO.<br></li>
<li><b>KernelSVM_SMO</b> - ядровой метод опорных векторов. Стандартный SMO метод.<br></li>
</ul>
