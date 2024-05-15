import numpy as np
import matplotlib.pyplot as plt

def calculate_a(b, K):
    return 1 / sum(np.exp(-b * i) for i in range(1, K + 1))

def f(x, a, b):
    return a * np.exp(-b * x)

K = int(input("K: "))
b = float(input("b: "))

a = calculate_a(b, K)

x = np.linspace(0, K, 400)

y = f(x, a, b)

x_int = np.arange(1, K + 1)
y_int = f(x_int, a, b)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'f(x) = {a:.3f} * exp(-{b} * x)')
plt.scatter(x_int, y_int, color='red') 
for i, j in zip(x_int, y_int):
    plt.text(i, j, f'({i},{j:.3f})', fontsize=9, ha='right')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = ae^{-bx}')
plt.legend()
plt.grid(True)
plt.show()
