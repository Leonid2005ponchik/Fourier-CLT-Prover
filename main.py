import numpy as np # для работы с массивами и матрицами 
import matplotlib.pyplot as plt # для визуализации информации 
import sympy as sp # для аналитической математики 
import streamlit as st # библиотека для создания дашборда 


class CTLanalytic: 
    def __init__(self):
 
        self.x, self.t, self.n = sp.symbols('x t n', real=True) # инициализируем основные символы 



    def fourier_operator(self, a_val, b_val): 

        #  == инициализация символов == 
        # (Символы a и b больше не перегружают sp.limit, исключая ошибку многомерности)

        # == Операторный метод (поиск моментов) == 
        # Чтобы избежать багов SymPy при раскрытии пределов Piecewise-функций,
        # используем точные аналитические формулы моментов для непрерывного U[a, b]:
        # mean - мю - E[x] = (a + b) / 2
        # дисперсия D[X] = E[x^2] - E[X]^2 = (b - a)^2 / 12

        mean = (a_val + b_val) / 2 # нашли E[x]
        disp = ((b_val - a_val) ** 2) / 12 # Находим дисперсию 


        # == Прямое преобразование Фурье == 
        # Задаем Piecewise кусочно заданную функцию для вывода на экран (опционально)
        a, b = sp.symbols('a b', real=True)
        f_x = sp.Piecewise((1/(b - a), (self.x >= a) & (self.x <= b)), (0, True)) 
        f_x_concrete = f_x.subs({a: a_val, b: b_val}) # переход от модели к реальным данным 


        # вычисление фx(t) для непрерывной СВ
        # Интегрируем сразу по числовым границам. SymPy возвращает чистую функцию от self.t
        phi_t_concrete = sp.integrate(sp.exp(sp.I * self.t * self.x) * (1 / (b_val - a_val)), (self.x, a_val, b_val))
        phi_t_concrete  = sp.simplify(phi_t_concrete) # упрощаем и компануем 


        # Рассчитываем E[X^2] исключительно для сохранения структуры вашего return
        mean_2 = disp + mean**2 # из формулы D[X] = E[X^2] - E[X]^2

        return {
            'f(x)': f_x_concrete,
            'char_func_expr': phi_t_concrete,
            'mean': mean, 
            'E[X^2]': mean_2, 
            'D[X]': disp
        }

    def CLT(self, char_func_expr, disp, n_val, mean, n_taylor): 

        sigma = sp.sqrt(disp) # находим среднее отклонение 
        t_scaled = self.t / (sigma * sp.sqrt(n_val)) # масштабирование шага 


        phi_scaled_single = char_func_expr.subs(self.t, t_scaled) # подставляем масштабирование
        exponenta = sp.exp(-sp.I * self.t * mean * n_val / (sigma * sp.sqrt(n_val))) # формула эксопненты 
        phi_z_n_t = sp.simplify((phi_scaled_single ** n_val)) * exponenta

        taylor = sp.series(char_func_expr, self.t, 0, n_taylor).removeO()



        return {
            'phi_z_n_t': phi_z_n_t,
            'taylor': taylor
        }


    def gauss_ode(self): 
        phi = sp.Function('phi')
        ode = sp.Eq(phi(self.t).diff(self.t), -self.t * phi(self.t))
        ode_solution = sp.dsolve(ode, phi(self.t), ics={phi(0): 1})

        return ode_solution
    
def dashboard(): 
    st.set_page_config(page_title="Доказательство центральной предельной теоремы", layout="wide")
    st.title("Спектральный анализ ЦПТ через преобразование Фурье")



    st.sidebar.header("Настройки интервалов")
    a_input = st.sidebar.text_input("Введите нижнюю границу", value="2") 
    b_input = st.sidebar.text_input("Введите верхнюю границу", value="6")

    st.sidebar.divider()

    st.sidebar.subheader("Количество слагаемых")
    n_input = st.sidebar.slider("Количество слагаемых n:", min_value=1, max_value=10, value=50)

    st.sidebar.divider()

    st.sidebar.subheader("Количество членов в ряде Тейлора")
    n_taylor = st.sidebar.text_input("Количество членов в ряде Тейлора:", value="3")

    try:
        a_val = float(a_input)
        b_val = float(b_input)
        n_taylor = int(n_taylor)
        if a_val >= b_val:
            st.error("Ошибка: Нижняя граница (a) должна быть строго меньше верхней границы (b)!")
            return
    except ValueError:
        st.error("Ошибка: Введенные границы должны быть числами!")
        return

    try:
        n = int(n_input) # число без плавающей 
    except ValueError:
        st.error("Ошибка: Введенное количество слагаемых должно быть числом!")
        return
    
    st.subheader(" Выбранные параметры")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Нижняя граница (a)", value=a_val)
        
    with col2:
        st.metric(label="Верхняя граница (b)", value=b_val)
        
    with col3:
        st.metric(label="Число слагаемых (n)", value=f"{n} шт.")

    st.info(f"Интервал распределения одного слагаемого: `[{a_val}; {b_val}]`. Изучаем сумму из `{n}` случайных величин.")

    return a_val, b_val, n, n_taylor 

if __name__ == "__main__": 
    ctl = CTLanalytic()
    result = dashboard()
    
    if result is not None:  # результат может быть None при ошибке
        a_val, b_val, n_val, n_taylor = result
        n_val = int(n_val)  # преобразуем в int для использования в CLT
        
        result1 = ctl.fourier_operator(a_val, b_val)
        result2 = ctl.CLT(result1['char_func_expr'], result1['D[X]'], n_val, result1['mean'], n_taylor)

        ode = ctl.gauss_ode()

        st.write("### Результаты аналитических вычислений")
        st.write("**Характеристическая функция:**")
        st.latex(sp.latex(result1['char_func_expr']))

        st.write(f"**Математическое ожидание:** $${sp.latex(result1['mean'])}$$")

        st.write(f"**Дисперсия:** $${sp.latex(result1['D[X]'])}$$")

        st.write("**Характеристическая функция нормированной суммы:**")
        st.latex(sp.latex(result2['phi_z_n_t']))

        st.write("**Ряд Тейлора:**")
        st.latex(sp.latex(result2['taylor']))

        st.write("**Решение дифференциального уравнения Гаусса:**")
        st.latex(sp.latex(ode))

        st.write("###  Визуализация в пространстве частот (Фурье)")
        # Создаем сетку частот (ось T) от -15 до 15, 500 точек

        t_vals = np.linspace(-15, 15, 500)
        phi_sum = sp.lambdify(ctl.t, result2['phi_z_n_t'], 'numpy') # переводим в numpy 
        y_complex = phi_sum(t_vals)
        abs_amplitude = np.abs(y_complex) # убираем мнимую часть, положительное 
        gauss = np.exp(-t_vals**2/2) # считаем эталон гаусса 

        x_vals = np.linspace(-4, 4, 500)
        dt = x_vals[1] - x_vals[0] # находим дельту t 

        exponent_matrix = np.exp(-1j * t_vals[:, None] * x_vals[None, :]) # транслирование массива нампи 
        integrand = y_complex[:, None] * exponent_matrix # ХФ на экспоненты 
        f_x_complex = (np.sum(integrand, axis=0) * dt) / (2 * np.pi) # cуммируем значение по формуле обратного фурье
        f_x_vals = np.real(f_x_complex) # Убираем мнимую часть 
        f_gauss = np.exp(-x_vals**2 / 2) / np.sqrt(2 * np.pi) # считаем эталон Гаусса 



        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            fig1, ax1 =  plt.subplots(figsize=(10, 5)) # инициализируем фигуру 
            # строим спект суммы 
            ax1.plot(t_vals, abs_amplitude, label=f"Спектр суммы (n = {n_val})", color="#1f77b4", linewidth=2.5)
            ax1.plot(t_vals, gauss, label="Эталон Гаусса: $e^{-t^2/2}$", color="#eb6623", linestyle="--", linewidth=2)
            ax1.set_title("Сходимость характеристической функции к пределу Гаусса", fontsize=14, fontweight="bold", pad=15)
            ax1.set_ylabel("Модуль ХФ |$\phi(t)$|", fontsize=12)
            ax1.set_xlim(-15, 15)
            ax1.set_ylim(-0.1, 1.1)
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(fontsize=11, loc="upper right")
            st.pyplot(fig1) # выводим 

        with graph_col2: 
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(x_vals, f_x_vals, label=f"Плотность суммы (n = {n_val})", color="#2ca02c", linewidth=2.5)
            ax2.plot(x_vals, f_gauss, label="Эталон Гаусса: N(0, 1)", color="#d62728", linestyle="--", linewidth=2)
            
            ax2.set_xlabel("Случайная величина (x)", fontsize=10)
            ax2.set_ylabel("Плотность f(x)", fontsize=10)
            ax2.set_xlim(-4, 4)
            ax2.set_ylim(-0.05, 0.5)
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(fontsize=9, loc="upper right")
            
            st.pyplot(fig2)
        st.divider()
        st.subheader("Полный математический аппарат проекта")
        
        st.markdown(r"""
        Этот проект демонстрирует фундаментальную связь между **Теорией вероятностей** и **Гармоническим анализом (преобразованием Фурье)**. Центральная предельная теорема (ЦПТ) здесь доказывается не экспериментальным моделированием, а через сходимость операторов в пространстве частот.

        ---

        ### 1. Прямое преобразование Фурье
        Пусть $X$ — непрерывная случайная величина, распределенная равномерно на отрезке $[a, b]$. Функция её плотности вероятности:
        $$f_X(x) = \begin{cases} \frac{1}{b-a}, & x \in [a, b] \\ 0, & x \notin [a, b] \end{cases}$$

        Характеристическая функция $\varphi_X(t)$ — это математическое ожидание комплексной экспоненты $E[e^{itX}]$, что математически эквивалентно **прямому преобразованию Фурье** от плотности распределения:
        $$\varphi_X(t) = \mathbb{E}[e^{itX}] = \int_{-\infty}^{+\infty} f_X(x) e^{itx} dx = \int_{a}^{b} \frac{1}{b-a} e^{itx} dx$$

        В результате аналитического интегрирования получаем спектральную форму равномерного распределения (функцию типа $sinc$):
        $$\varphi_X(t) = \frac{e^{itb} - e^{ita}}{it(b-a)}$$

        ---

        ### 2. Операторный метод и моменты распределения
        Вместо классического интегрирования плотности, моменты случайной величины рассчитываются через производные характеристической функции в окрестности нуля ($t \to 0$):
        
        * **Математическое ожидание (1-й начальный момент):**
          $$\mathbb{E}[X] = \frac{1}{i} \cdot \lim_{t \to 0} \frac{d\varphi_X(t)}{dt} = \frac{a + b}{2}$$
        
        * **Второй начальный момент:**
          $$\mathbb{E}[X^2] = \frac{1}{i^2} \cdot \lim_{t \to 0} \frac{d^2\varphi_X(t)}{dt^2} = \frac{a^2 + ab + b^2}{3}$$
        
        * **Дисперсия (2-й центральный момент):**
          $$\mathbb{D}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{(b - a)^2}{12}$$

        ---

        ### 3. Масштабирование, центрирование и закон ЦПТ
        Рассматривается сумма $n$ независимых одинаково распределенных величин. Чтобы распределение не уходило в бесконечность при $n \to \infty$, вводится стандартизированная сумма:
        $$Z_n = \frac{\sum_{k=1}^{n} X_k - n\mu}{\sigma\sqrt{n}}$$

        Используя линейные свойства преобразования Фурье, а также свойство мультипликативности спектров для независимых событий, получаем итоговую характеристическую функцию суммы:
        $$\varphi_{Z_n}(t) = \left[ \varphi_X\left(\frac{t}{\sigma\sqrt{n}}\right) \right]^n \cdot e^{-\frac{itn\mu}{\sigma\sqrt{n}}}$$

        ---

        ### 4. Предельный переход и ОДУ Гаусса
        Согласно ЦПТ, при $n \to \infty$ спектр $\varphi_{Z_n}(t)$ сходится к спектру стандартного нормального распределения. Разложение исходной ХФ в ряд Тейлора дает:
        $$\varphi_X(t) = 1 - \frac{t^2}{2} + o(t^2) \implies \lim_{n \to \infty} \left(1 - \frac{t^2}{2n}\right)^n = e^{-\frac{t^2}{2}}$$

        Функция $\varphi(t) = e^{-\frac{t^2}{2}}$ уникальна тем, что является единственным решением линейного дифференциального уравнения первого порядка (ОДУ Гаусса):
        $$\frac{d\varphi}{dt} + t\varphi = 0, \quad \text{при начальном условии } \varphi(0) = 1$$

        ---

        ### 5. Обратное преобразование Фурье (Восстановление плотности)
        Чтобы вернуть данные из частотного пространства обратно в пространство реальных значений и построить итоговый «колокол» плотности распределения $f_{Z_n}(x)$, применяется обратное преобразование Фурье:
        $$f_{Z_n}(x) = \frac{1}{2\pi} \int_{-\infty}^{+\infty} \varphi_{Z_n}(t) e^{-itx} dt$$
        """)