import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import symbols, lambdify, sympify, diff

class NumericalMethodsCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Métodos Numéricos")
        self.root.geometry("1000x700")
        
        self.create_widgets()
        self.setup_plots()
        
    def create_widgets(self):
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Métodos cerrados
        self.create_closed_methods_tab()
        # Métodos abiertos
        self.create_open_methods_tab()
        # Sistemas de ecuaciones
        self.create_linear_systems_tab()
        # Interpolación
        self.create_interpolation_tab()
        # Runge-Kutta
        self.create_runge_kutta_tab()
        
    def setup_plots(self):
        # Configuración de figuras para cada pestaña
        self.fig_closed = plt.figure(figsize=(6, 4))
        self.canvas_closed = FigureCanvasTkAgg(self.fig_closed, master=self.closed_methods_tab)
        self.canvas_closed.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig_open = plt.figure(figsize=(6, 4))
        self.canvas_open = FigureCanvasTkAgg(self.fig_open, master=self.open_methods_tab)
        self.canvas_open.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig_linear = plt.figure(figsize=(6, 4))
        self.canvas_linear = FigureCanvasTkAgg(self.fig_linear, master=self.linear_systems_tab)
        self.canvas_linear.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig_interp = plt.figure(figsize=(6, 4))
        self.canvas_interp = FigureCanvasTkAgg(self.fig_interp, master=self.interpolation_tab)
        self.canvas_interp.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig_rk = plt.figure(figsize=(6, 4))
        self.canvas_rk = FigureCanvasTkAgg(self.fig_rk, master=self.runge_kutta_tab)
        self.canvas_rk.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
    def create_closed_methods_tab(self):
        self.closed_methods_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.closed_methods_tab, text="Métodos Cerrados")
        
        # Frame para controles
        control_frame = ttk.LabelFrame(self.closed_methods_tab, text="Parámetros")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Función
        ttk.Label(control_frame, text="Función f(x):").grid(row=0, column=0, padx=5, pady=5)
        self.func_entry_closed = ttk.Entry(control_frame, width=30)
        self.func_entry_closed.grid(row=0, column=1, padx=5, pady=5)
        self.func_entry_closed.insert(0, "x**3 - 2*x - 5")
        
        # Intervalo [a, b]
        ttk.Label(control_frame, text="Intervalo [a, b]:").grid(row=1, column=0, padx=5, pady=5)
        self.a_entry = ttk.Entry(control_frame, width=10)
        self.a_entry.grid(row=1, column=1, padx=5, pady=5)
        self.a_entry.insert(0, "1")
        
        self.b_entry = ttk.Entry(control_frame, width=10)
        self.b_entry.grid(row=1, column=2, padx=5, pady=5)
        self.b_entry.insert(0, "3")
        
        # Tolerancia
        ttk.Label(control_frame, text="Tolerancia:").grid(row=2, column=0, padx=5, pady=5)
        self.tol_entry_closed = ttk.Entry(control_frame, width=10)
        self.tol_entry_closed.grid(row=2, column=1, padx=5, pady=5)
        self.tol_entry_closed.insert(0, "0.0001")
        
        # Máximo de iteraciones
        ttk.Label(control_frame, text="Max iteraciones:").grid(row=3, column=0, padx=5, pady=5)
        self.max_iter_entry_closed = ttk.Entry(control_frame, width=10)
        self.max_iter_entry_closed.grid(row=3, column=1, padx=5, pady=5)
        self.max_iter_entry_closed.insert(0, "100")
        
        # Mostrar iteraciones
        ttk.Label(control_frame, text="Mostrar iteraciones:").grid(row=4, column=0, padx=5, pady=5)
        self.show_iter_closed = tk.IntVar(value=1)
        ttk.Checkbutton(control_frame, variable=self.show_iter_closed).grid(row=4, column=1, padx=5, pady=5)
        
        # Botones de métodos
        method_frame = ttk.Frame(control_frame)
        method_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        ttk.Button(method_frame, text="Bisección", command=self.bisection_method).pack(side=tk.LEFT, padx=5)
        ttk.Button(method_frame, text="Falsa Posición", command=self.false_position_method).pack(side=tk.LEFT, padx=5)
        
        # Resultados
        self.result_text_closed = tk.Text(self.closed_methods_tab, height=10, width=80)
        self.result_text_closed.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_open_methods_tab(self):
        self.open_methods_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.open_methods_tab, text="Métodos Abiertos")
        
        # Frame para controles
        control_frame = ttk.LabelFrame(self.open_methods_tab, text="Parámetros")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Función
        ttk.Label(control_frame, text="Función f(x):").grid(row=0, column=0, padx=5, pady=5)
        self.func_entry_open = ttk.Entry(control_frame, width=30)
        self.func_entry_open.grid(row=0, column=1, padx=5, pady=5)
        self.func_entry_open.insert(0, "x**3 - 2*x - 5")
        
        # Valor inicial
        ttk.Label(control_frame, text="Valor inicial (x0):").grid(row=1, column=0, padx=5, pady=5)
        self.x0_entry = ttk.Entry(control_frame, width=10)
        self.x0_entry.grid(row=1, column=1, padx=5, pady=5)
        self.x0_entry.insert(0, "2")
        
        # Para Secante: x1
        ttk.Label(control_frame, text="x1 (para Secante):").grid(row=2, column=0, padx=5, pady=5)
        self.x1_entry = ttk.Entry(control_frame, width=10)
        self.x1_entry.grid(row=2, column=1, padx=5, pady=5)
        self.x1_entry.insert(0, "3")
        
        # Tolerancia
        ttk.Label(control_frame, text="Tolerancia:").grid(row=3, column=0, padx=5, pady=5)
        self.tol_entry_open = ttk.Entry(control_frame, width=10)
        self.tol_entry_open.grid(row=3, column=1, padx=5, pady=5)
        self.tol_entry_open.insert(0, "0.0001")
        
        # Máximo de iteraciones
        ttk.Label(control_frame, text="Max iteraciones:").grid(row=4, column=0, padx=5, pady=5)
        self.max_iter_entry_open = ttk.Entry(control_frame, width=10)
        self.max_iter_entry_open.grid(row=4, column=1, padx=5, pady=5)
        self.max_iter_entry_open.insert(0, "100")
        
        # Mostrar iteraciones
        ttk.Label(control_frame, text="Mostrar iteraciones:").grid(row=5, column=0, padx=5, pady=5)
        self.show_iter_open = tk.IntVar(value=1)
        ttk.Checkbutton(control_frame, variable=self.show_iter_open).grid(row=5, column=1, padx=5, pady=5)
        
        # Botones de métodos
        method_frame = ttk.Frame(control_frame)
        method_frame.grid(row=6, column=0, columnspan=3, pady=10)
        
        ttk.Button(method_frame, text="Newton-Raphson", command=self.newton_raphson_method).pack(side=tk.LEFT, padx=5)
        ttk.Button(method_frame, text="Secante", command=self.secant_method).pack(side=tk.LEFT, padx=5)
        
        # Resultados
        self.result_text_open = tk.Text(self.open_methods_tab, height=10, width=80)
        self.result_text_open.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_linear_systems_tab(self):
        self.linear_systems_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.linear_systems_tab, text="Sistemas Lineales")
        
        # Frame para controles
        control_frame = ttk.LabelFrame(self.linear_systems_tab, text="Gauss-Seidel")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Matriz A
        ttk.Label(control_frame, text="Matriz A (separar filas con ; y elementos con ,):").grid(row=0, column=0, padx=5, pady=5)
        self.matrix_a_entry = ttk.Entry(control_frame, width=40)
        self.matrix_a_entry.grid(row=0, column=1, padx=5, pady=5)
        self.matrix_a_entry.insert(0, "4, -1, 0, 3; 1, 15.5, 3, 8; 0, -1.3, -4, 1.1; 14, 5, -2, 30")
        
        # Vector b
        ttk.Label(control_frame, text="Vector b (separar elementos con ,):").grid(row=1, column=0, padx=5, pady=5)
        self.vector_b_entry = ttk.Entry(control_frame, width=40)
        self.vector_b_entry.grid(row=1, column=1, padx=5, pady=5)
        self.vector_b_entry.insert(0, "1, 1, 1, 1")
        
        # Vector inicial
        ttk.Label(control_frame, text="Vector inicial (separar elementos con ,):").grid(row=2, column=0, padx=5, pady=5)
        self.initial_vector_entry = ttk.Entry(control_frame, width=40)
        self.initial_vector_entry.grid(row=2, column=1, padx=5, pady=5)
        self.initial_vector_entry.insert(0, "0, 0, 0, 0")
        
        # Tolerancia
        ttk.Label(control_frame, text="Tolerancia:").grid(row=3, column=0, padx=5, pady=5)
        self.tol_entry_linear = ttk.Entry(control_frame, width=10)
        self.tol_entry_linear.grid(row=3, column=1, padx=5, pady=5)
        self.tol_entry_linear.insert(0, "0.0001")
        
        # Máximo de iteraciones
        ttk.Label(control_frame, text="Max iteraciones:").grid(row=4, column=0, padx=5, pady=5)
        self.max_iter_entry_linear = ttk.Entry(control_frame, width=10)
        self.max_iter_entry_linear.grid(row=4, column=1, padx=5, pady=5)
        self.max_iter_entry_linear.insert(0, "100")
        
        # Botón de método
        ttk.Button(control_frame, text="Resolver con Gauss-Seidel", command=self.gauss_seidel_method).grid(row=5, column=0, columnspan=2, pady=10)
        
        # Resultados
        self.result_text_linear = tk.Text(self.linear_systems_tab, height=10, width=80)
        self.result_text_linear.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        pass
    
    def create_interpolation_tab(self):
        self.interpolation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.interpolation_tab, text="Interpolación")
        
        # Frame para controles
        control_frame = ttk.LabelFrame(self.interpolation_tab, text="Parámetros")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Puntos x
        ttk.Label(control_frame, text="Puntos x (separar con ,):").grid(row=0, column=0, padx=5, pady=5)
        self.x_points_entry = ttk.Entry(control_frame, width=30)
        self.x_points_entry.grid(row=0, column=1, padx=5, pady=5)
        self.x_points_entry.insert(0, "1, 2, 3, 4")
        
        # Puntos y
        ttk.Label(control_frame, text="Puntos y (separar con ,):").grid(row=1, column=0, padx=5, pady=5)
        self.y_points_entry = ttk.Entry(control_frame, width=30)
        self.y_points_entry.grid(row=1, column=1, padx=5, pady=5)
        self.y_points_entry.insert(0, "1, 4, 9, 16")
        
        # Punto a interpolar
        ttk.Label(control_frame, text="Punto a interpolar (x):").grid(row=2, column=0, padx=5, pady=5)
        self.interp_point_entry = ttk.Entry(control_frame, width=10)
        self.interp_point_entry.grid(row=2, column=1, padx=5, pady=5)
        self.interp_point_entry.insert(0, "2.5")
        
        # Botones de métodos
        method_frame = ttk.Frame(control_frame)
        method_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(method_frame, text="Polinomio de Newton", command=self.newton_polynomial).pack(side=tk.LEFT, padx=5)
        ttk.Button(method_frame, text="Polinomio de Lagrange", command=self.lagrange_polynomial).pack(side=tk.LEFT, padx=5)
        
        # Resultados
        self.result_text_interp = tk.Text(self.interpolation_tab, height=10, width=80)
        self.result_text_interp.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        pass
    
    def create_runge_kutta_tab(self):
        self.runge_kutta_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.runge_kutta_tab, text="Runge-Kutta")
        
        # Frame para controles
        control_frame = ttk.LabelFrame(self.runge_kutta_tab, text="Parámetros")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Ecuación diferencial
        ttk.Label(control_frame, text="dy/dx = f(x, y):").grid(row=0, column=0, padx=5, pady=5)
        self.ode_entry = ttk.Entry(control_frame, width=30)
        self.ode_entry.grid(row=0, column=1, padx=5, pady=5)
        self.ode_entry.insert(0, "x + y")
        
        # Condición inicial
        ttk.Label(control_frame, text="y(x0) = y0:").grid(row=1, column=0, padx=5, pady=5)
        
        self.x0_rk_entry = ttk.Entry(control_frame, width=10)
        self.x0_rk_entry.grid(row=1, column=1, padx=5, pady=5)
        self.x0_rk_entry.insert(0, "0")
        
        self.y0_rk_entry = ttk.Entry(control_frame, width=10)
        self.y0_rk_entry.grid(row=1, column=2, padx=5, pady=5)
        self.y0_rk_entry.insert(0, "1")
        
        # Punto final
        ttk.Label(control_frame, text="Punto final (xf):").grid(row=2, column=0, padx=5, pady=5)
        self.xf_entry = ttk.Entry(control_frame, width=10)
        self.xf_entry.grid(row=2, column=1, padx=5, pady=5)
        self.xf_entry.insert(0, "1")
        
        # Paso (h)
        ttk.Label(control_frame, text="Tamaño de paso (h):").grid(row=3, column=0, padx=5, pady=5)
        self.h_entry = ttk.Entry(control_frame, width=10)
        self.h_entry.grid(row=3, column=1, padx=5, pady=5)
        self.h_entry.insert(0, "0.1")
        
        # Botón de método
        ttk.Button(control_frame, text="Resolver con Runge-Kutta 4to orden", command=self.runge_kutta_4).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Resultados
        self.result_text_rk = tk.Text(self.runge_kutta_tab, height=10, width=80)
        self.result_text_rk.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        pass

    # ==================== MÉTODOS NUMÉRICOS ====================
    
    def bisection_method(self):
        try:
            # Obtener parámetros
            func_str = self.func_entry_closed.get()
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            tol = float(self.tol_entry_closed.get())
            max_iter = int(self.max_iter_entry_closed.get())
            show_iter = self.show_iter_closed.get()
            
            # Definir la función
            x = symbols('x')
            try:
                f_expr = sympify(func_str)
                f = lambdify(x, f_expr, 'numpy')
            except:
                messagebox.showerror("Error", "Función no válida")
                return
            
            # Verificar que f(a) y f(b) tengan signos opuestos
            fa = f(a)
            fb = f(b)
            
            if fa * fb >= 0:
                messagebox.showerror("Error", "La función debe tener signos opuestos en los extremos del intervalo")
                return
            
            # Realizar el método de bisección
            results = []
            iterations = []
            for i in range(max_iter):
                c = (a + b) / 2
                fc = f(c)
                
                iterations.append((i+1, a, b, c, fc))
                
                if show_iter:
                    results.append(
                        f"Iteración {i+1}:\n"
                        f" X{i} -> (c) = {c:.8f} (Punto Medio entre a: {a:.6f} y b: {b:.6f})\n"
                        f" f(X{i}) = {fc:.8f}\n"
                        f" Error: {abs(b-a)/2:.8f}")
                
                if abs(fc) < tol:
                    break
                
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            
            # Mostrar resultados
            self.result_text_closed.delete(1.0, tk.END)
            self.result_text_closed.insert(tk.END, f"Solución encontrada: x = {c:.8f}\n\n")
            
            if show_iter:
                self.result_text_closed.insert(tk.END, "\n".join(results))
            
            # Graficar
            self.plot_function(f, a, b, c, "Método de Bisección", iterations)
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def false_position_method(self):
        try:
            # Obtener parámetros
            func_str = self.func_entry_closed.get()
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            tol = float(self.tol_entry_closed.get())
            max_iter = int(self.max_iter_entry_closed.get())
            show_iter = self.show_iter_closed.get()
            
            # Definir la función
            x = symbols('x')
            try:
                f_expr = sympify(func_str)
                f = lambdify(x, f_expr, 'numpy')
            except:
                messagebox.showerror("Error", "Función no válida")
                return
            
            # Verificar que f(a) y f(b) tengan signos opuestos
            fa = f(a)
            fb = f(b)
            
            if fa * fb >= 0:
                messagebox.showerror("Error", "La función debe tener signos opuestos en los extremos del intervalo")
                return
            
            # Realizar el método de falsa posición
            results = []
            iterations = []
            for i in range(max_iter):
                c = (a * fb - b * fa) / (fb - fa)
                fc = f(c)
                
                iterations.append((i+1, a, b, c, fc))
                
                if show_iter:
                    results.append(f"Iteración {i+1}: a = {a:.6f}, b = {b:.6f}, c = {c:.6f}, f(c) = {fc:.6f}")
                
                if abs(fc) < tol:
                    break
                
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            
            # Mostrar resultados
            self.result_text_closed.delete(1.0, tk.END)
            self.result_text_closed.insert(tk.END, f"Solución encontrada: x = {c:.8f}\n\n")
            
            if show_iter:
                self.result_text_closed.insert(tk.END, "\n".join(results))
            
            # Graficar
            self.plot_function(f, a, b, c, "Método de Falsa Posición", iterations)
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def newton_raphson_method(self):
        try:
            # Obtener parámetros
            func_str = self.func_entry_open.get()
            x0 = float(self.x0_entry.get())
            tol = float(self.tol_entry_open.get())
            max_iter = int(self.max_iter_entry_open.get())
            show_iter = self.show_iter_open.get()
            
            # Definir la función y su derivada
            x = symbols('x')
            try:
                f_expr = sympify(func_str)
                f = lambdify(x, f_expr, 'numpy')
                df_expr = diff(f_expr, x)
                df = lambdify(x, df_expr, 'numpy')
            except:
                messagebox.showerror("Error", "Función no válida")
                return
            
            # Realizar el método de Newton-Raphson
            results = []
            iterations = []
            x_vals = [x0]
            
            for i in range(max_iter):
                fx = f(x0)
                dfx = df(x0)
                
                if dfx == 0:
                    messagebox.showerror("Error", "Derivada cero. No se puede continuar.")
                    return
                
                x1 = x0 - fx / dfx
                error = abs(x1 - x0)
                
                iterations.append((i+1, x0, x1, fx, dfx, error))
                x_vals.append(x1)
                
                if show_iter:
                    results.append(f"Iteración {i+1}:\n"
                                   f" X{i} = {x0:.8f}\n"
                                   f" f(X{i}) = {fx:.8f}\n"
                                   f" f'(X{i}) = {dfx:.8f}\n"
                                   f" X{i+1} = {x1:.8f}\n"
                                   f" Error: {error:.8f}")
                
                if error < tol:
                    break
                
                x0 = x1
            
            # Mostrar resultados
            self.result_text_open.delete(1.0, tk.END)
            self.result_text_open.insert(tk.END, f"Solución encontrada: x = {x0:.8f}\n\n")
            
            if show_iter:
                self.result_text_open.insert(tk.END, "\n".join(results))
            
            # Graficar
            self.plot_open_method(f, x_vals, "Método de Newton-Raphson", iterations)
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def secant_method(self):
        try:
            # Obtener parámetros
            func_str = self.func_entry_open.get()
            x0 = float(self.x0_entry.get())
            x1 = float(self.x1_entry.get())
            tol = float(self.tol_entry_open.get())
            max_iter = int(self.max_iter_entry_open.get())
            show_iter = self.show_iter_open.get()
            
            # Definir la función
            x = symbols('x')
            try:
                f_expr = sympify(func_str)
                f = lambdify(x, f_expr, 'numpy')
            except:
                messagebox.showerror("Error", "Función no válida")
                return
            # Realizar el método de la secante
            results = []
            iterations = []
            x_vals = [x0, x1]
            
            for i in range(max_iter):
                fx0 = f(x0)
                fx1 = f(x1)
                
                if fx1 - fx0 == 0:
                    messagebox.showerror("Error", "División por cero. No se puede continuar.")
                    return
                
                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                fx2 = f(x2)
                error = abs(x2 - x1)
                
                iterations.append((i+1, x0, x1, x2, fx0, fx1, fx2, error))
                x_vals.append(x2)
                if show_iter:
                    results.append(f"Iteración {i+1}:\n x{i} = {x0:.8f}, x{i+1} = {x1:.8f}:\n"
                                   f" f{i} = {fx0:.8f}, f{i+1} = {fx1:.8f}\n"
                                   f" x{i+2} = {x2:.8f}, error = {error:.8f}")
                    #results.append(f"Iteración {i+1}:\n x0 = {x0:.8f}, x1 = {x1:.8f}, x2 = {x2:.8f}, f(x2) = {fx2:.8f}, error = {error:.8f}")
                    if error < tol:
                        break
                    x0, x1 = x1, x2
            
            # Mostrar resultados
            self.result_text_open.delete(1.0, tk.END)
            self.result_text_open.insert(tk.END, f"Solución encontrada: x = {x1:.8f}\n\n")
            
            if show_iter:
                self.result_text_open.insert(tk.END, "\n".join(results))
                
                # Graficar
                self.plot_open_method(f, x_vals, "Método de la Secante", iterations)
                
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def gauss_seidel_method(self):
        try:
            # Obtener parámetros
            matrix_a_str = self.matrix_a_entry.get()
            vector_b_str = self.vector_b_entry.get()
            initial_vector_str = self.initial_vector_entry.get()
            tol = float(self.tol_entry_linear.get())
            max_iter = int(self.max_iter_entry_linear.get())
            
            # Procesar matriz A
            rows = matrix_a_str.split(';')
            A = []
            for row in rows:
                elements = [float(x.strip()) for x in row.split(',')]
                A.append(elements)
            A = np.array(A, dtype=float)
            
            # Procesar vector b
            b = np.array([float(x.strip()) for x in vector_b_str.split(',')], dtype=float)
            
            # Procesar vector inicial
            x = np.array([float(x.strip()) for x in initial_vector_str.split(',')], dtype=float)
            
            n = len(b)
            results = []
            
            # Verificar convergencia (matriz diagonal dominante)
            for i in range(n):
                diagonal = abs(A[i,i])
                row_sum = np.sum(np.abs(A[i,:])) - diagonal
                if diagonal <= row_sum:
                    messagebox.showwarning("Advertencia", "La matriz puede no ser diagonal dominante. La convergencia no está garantizada.")
                    break
            
            # Método de Gauss-Seidel
            for k in range(max_iter):
                x_old = x.copy()
                for i in range(n):
                    s1 = np.dot(A[i,:i], x[:i])
                    s2 = np.dot(A[i,i+1:], x_old[i+1:])
                    x[i] = (b[i] - s1 - s2) / A[i,i]
                
                error = np.linalg.norm(x - x_old, np.inf)
                results.append(f"Iteración {k+1}: x = {x}, error = {error:.8f}")
                
                if error < tol:
                    break
            
            # Mostrar resultados
            self.result_text_linear.delete(1.0, tk.END)
            self.result_text_linear.insert(tk.END, f"Solución encontrada:\n{x}\n\n")
            self.result_text_linear.insert(tk.END, "\n".join(results))
            
            # Graficar (para sistemas 2x2 o 3x3)
            if n == 2:
                self.plot_linear_system_2d(A, b, x)
            elif n == 3:
                self.plot_linear_system_3d(A, b, x)
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def newton_polynomial(self):
        try:
            # Obtener puntos
            x_points = np.array([float(x.strip()) for x in self.x_points_entry.get().split(',')])
            y_points = np.array([float(y.strip()) for y in self.y_points_entry.get().split(',')])
            x_interp = float(self.interp_point_entry.get())
            
            if len(x_points) != len(y_points):
                messagebox.showerror("Error", "El número de puntos x debe ser igual al número de puntos y")
                return
            
            n = len(x_points)
            
            # Calcular diferencias divididas
            F = np.zeros((n, n))
            F[:,0] = y_points
            
            for j in range(1, n):
                for i in range(n - j):
                    F[i,j] = (F[i+1,j-1] - F[i,j-1]) / (x_points[i+j] - x_points[i])
            
            # Construir el polinomio
            result = f"Coeficientes del polinomio de Newton:\n{F[0]}\n\n"
            result += f"Polinomio:\nP(x) = {F[0,0]:.4f}"
            
            for i in range(1, n):
                term = f" + {F[0,i]:.4f}"
                for j in range(i):
                    term += f"(x - {x_points[j]:.2f})"
                result += term
            
            # Evaluar en el punto deseado
            P = F[0,0]
            product_terms = 1.0
            
            for i in range(1, n):
                product_terms *= (x_interp - x_points[i-1])
                P += F[0,i] * product_terms
            
            result += f"\n\nP({x_interp}) = {P:.8f}"
            
            # Mostrar resultados
            self.result_text_interp.delete(1.0, tk.END)
            self.result_text_interp.insert(tk.END, result)
            
            # Graficar
            self.plot_interpolation(x_points, y_points, x_interp, P, "Polinomio de Newton")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def lagrange_polynomial(self):
        try:
            # Obtener puntos
            x_points = np.array([float(x.strip()) for x in self.x_points_entry.get().split(',')])
            y_points = np.array([float(y.strip()) for y in self.y_points_entry.get().split(',')])
            x_interp = float(self.interp_point_entry.get())
            
            if len(x_points) != len(y_points):
                messagebox.showerror("Error", "El número de puntos x debe ser igual al número de puntos y")
                return
            
            n = len(x_points)
            P = 0.0
            result = "Polinomio de Lagrange:\nP(x) = "
            
            # Calcular polinomio de Lagrange
            terms = []
            for i in range(n):
                term = f"{y_points[i]:.4f}"
                denominator = 1.0
                for j in range(n):
                    if j != i:
                        term += f" * (x - {x_points[j]:.2f})"
                        denominator *= (x_points[i] - x_points[j])
                term = f"({term}) / {denominator:.4f}"
                terms.append(term)
                
                # Evaluar en x_interp
                product = y_points[i]
                for j in range(n):
                    if j != i:
                        product *= (x_interp - x_points[j]) / (x_points[i] - x_points[j])
                P += product
            
            result += " + ".join(terms)
            result += f"\n\nP({x_interp}) = {P:.8f}"
            
            # Mostrar resultados
            self.result_text_interp.delete(1.0, tk.END)
            self.result_text_interp.insert(tk.END, result)
            
            # Graficar
            self.plot_interpolation(x_points, y_points, x_interp, P, "Polinomio de Lagrange")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def runge_kutta_4(self):
        try:
            # Obtener parámetros
            ode_str = self.ode_entry.get()
            x0 = float(self.x0_rk_entry.get())
            y0 = float(self.y0_rk_entry.get())
            xf = float(self.xf_entry.get())
            h = float(self.h_entry.get())
            
            # Definir la función ODE
            x, y = symbols('x y')
            try:
                f_expr = sympify(ode_str)
                f = lambdify((x, y), f_expr, 'numpy')
            except:
                messagebox.showerror("Error", "Ecuación diferencial no válida")
                return
            
            # Preparar arrays para almacenar resultados
            num_steps = int((xf - x0) / h) + 1
            x_values = np.linspace(x0, xf, num_steps)
            y_values = np.zeros(num_steps)
            y_values[0] = y0
            
            # Método de Runge-Kutta 4to orden
            results = []
            results.append(f"x = {x0:.4f}, y = {y0:.8f}")
            
            for i in range(1, num_steps):
                xi = x_values[i-1]
                yi = y_values[i-1]
                
                k1 = h * f(xi, yi)
                k2 = h * f(xi + h/2, yi + k1/2)
                k3 = h * f(xi + h/2, yi + k2/2)
                k4 = h * f(xi + h, yi + k3)
                
                y_values[i] = yi + (k1 + 2*k2 + 2*k3 + k4) / 6
                results.append(f"x = {x_values[i]:.4f}, y = {y_values[i]:.8f}")
            
            # Mostrar resultados
            self.result_text_rk.delete(1.0, tk.END)
            self.result_text_rk.insert(tk.END, "\n".join(results))
            
            # Graficar
            self.plot_runge_kutta(x_values, y_values)
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    # ==================== FUNCIONES AUXILIARES PARA GRAFICAR ====================
    
    def plot_function(self, f, a, b, root, title, iterations=None):
        self.fig_closed.clf()
        ax = self.fig_closed.add_subplot(111)
        
        # Generar puntos para graficar
        x_vals = np.linspace(a, b, 400)
        y_vals = f(x_vals)
        
        # Graficar función
        ax.plot(x_vals, y_vals, label='f(x)')
        
        # Graficar raíz encontrada
        ax.scatter([root], [f(root)], color='red', label=f'Raíz ≈ {root:.4f}')
        
        # Graficar iteraciones si se desea
        if iterations and len(iterations) > 0:
            for i, iter_data in enumerate(iterations):
                if len(iter_data) == 5:  # Para bisección y falsa posición
                    _, a_iter, b_iter, c_iter, fc_iter = iter_data
                    ax.scatter([c_iter], [fc_iter], color='green', alpha=0.3)
                    if i == len(iterations)-1:  # Última iteración
                        ax.plot([a_iter, b_iter], [f(a_iter), f(b_iter)], 'g--', alpha=0.3, label='Intervalos')
                    else:
                        ax.plot([a_iter, b_iter], [f(a_iter), f(b_iter)], 'g--', alpha=0.3)
        
        # Línea y=0
        ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True)
        
        self.canvas_closed.draw()

    def plot_open_method(self, f, x_vals, title, iterations=None):
        self.fig_open.clf()
        ax = self.fig_open.add_subplot(111)
        
        # Determinar rango para graficar
        min_x = min(x_vals)
        max_x = max(x_vals)
        padding = 0.5 * (max_x - min_x) if max_x != min_x else 1.0
        x_range = np.linspace(min_x - padding, max_x + padding, 400)
        
        # Graficar función
        ax.plot(x_range, f(x_range), label='f(x)')
        
        # Graficar puntos de iteración
        for i, x in enumerate(x_vals):
            if i == len(x_vals) - 1:  # Último punto (solución)
                ax.scatter([x], [f(x)], color='red', label=f'Solución ≈ {x:.4f}')
            else:
                ax.scatter([x], [f(x)], color='green', alpha=0.5)
                
        # Graficar líneas de iteración para Newton-Raphson
        if "Newton" in title and iterations:
            for i, iter_data in enumerate(iterations):
                if len(iter_data) == 6:  # Para Newton-Raphson
                    _, x0, x1, fx, dfx, error = iter_data
                    # Línea tangente
                    tangent_line = lambda x: fx + dfx * (x - x0)
                    ax.plot(x_range, tangent_line(x_range), 'g--', alpha=0.3)
                    # Línea vertical a x1
                    ax.plot([x1, x1], [0, f(x1)], 'r--', alpha=0.3)
        
        # Línea y=0
        ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True)
        
        self.canvas_open.draw()
    
    def plot_linear_system_2d(self, A, b, solution):
        self.fig_linear.clf()
        ax = self.fig_linear.add_subplot(111)
        
        # Para sistemas 2x2
        x = np.linspace(-10, 10, 400)
        
        # Graficar cada ecuación
        for i in range(2):
            if A[i,1] != 0:
                y = (b[i] - A[i,0] * x) / A[i,1]
                ax.plot(x, y, label=f'Ecuación {i+1}')
            else:  # línea vertical
                x_val = b[i] / A[i,0]
                ax.axvline(x=x_val, label=f'Ecuación {i+1}')
        
        # Graficar solución
        ax.scatter([solution[0]], [solution[1]], color='red', label=f'Solución ({solution[0]:.2f}, {solution[1]:.2f})')
        
        ax.set_title('Sistema de Ecuaciones Lineales (2D)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        ax.grid(True)
        
        self.canvas_linear.draw()
    
    def plot_linear_system_3d(self, A, b, solution):
        self.fig_linear.clf()
        ax = self.fig_linear.add_subplot(111, projection='3d')
        
        # Para sistemas 3x3
        x = np.linspace(-10, 10, 20)
        y = np.linspace(-10, 10, 20)
        X, Y = np.meshgrid(x, y)
        
        # Graficar cada ecuación
        for i in range(3):
            if A[i,2] != 0:
                Z = (b[i] - A[i,0] * X - A[i,1] * Y) / A[i,2]
                ax.plot_surface(X, Y, Z, alpha=0.5, label=f'Ecuación {i+1}')
        
        # Graficar solución
        ax.scatter([solution[0]], [solution[1]], [solution[2]], color='red', 
                  label=f'Solución ({solution[0]:.2f}, {solution[1]:.2f}, {solution[2]:.2f})')
        
        ax.set_title('Sistema de Ecuaciones Lineales (3D)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.legend()
        
        self.canvas_linear.draw()
    
    def plot_interpolation(self, x_points, y_points, x_interp, y_interp, title):
        self.fig_interp.clf()
        ax = self.fig_interp.add_subplot(111)
        
        # Graficar puntos originales
        ax.scatter(x_points, y_points, color='red', label='Puntos dados')
        
        # Graficar punto interpolado
        ax.scatter([x_interp], [y_interp], color='blue', label=f'P({x_interp:.2f}) = {y_interp:.4f}')
        
        # Crear una función de interpolación para graficar la curva
        if len(x_points) > 1:
            x_vals = np.linspace(min(x_points), max(x_points), 400)
            
            # Usar polinomio de Lagrange para graficar
            y_vals = np.zeros_like(x_vals)
            n = len(x_points)
            
            for i in range(n):
                term = y_points[i] * np.ones_like(x_vals)
                for j in range(n):
                    if j != i:
                        term *= (x_vals - x_points[j]) / (x_points[i] - x_points[j])
                y_vals += term
            
            ax.plot(x_vals, y_vals, label='Polinomio de interpolación')
        
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True)
        
        self.canvas_interp.draw()
    
    def plot_runge_kutta(self, x_values, y_values):
        self.fig_rk.clf()
        ax = self.fig_rk.add_subplot(111)
        
        ax.plot(x_values, y_values, 'b-', label='Solución numérica')
        ax.scatter(x_values, y_values, color='red', s=30)
        
        ax.set_title('Solución de EDO con Runge-Kutta 4to Orden')
        ax.set_xlabel('x')
        ax.set_ylabel('y(x)')
        ax.legend()
        ax.grid(True)
        
        self.canvas_rk.draw()

# Iniciar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = NumericalMethodsCalculator(root)
    root.mainloop()