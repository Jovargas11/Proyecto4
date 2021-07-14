#Importaciones necesarias
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image


# Función para cargar la imagen en un array
def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)

#Función para pasar de rgb a array
def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

#Solución ejercicio 4.1:

#Primero se va a crear un modulador para el caso 16-QAM

def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora_b1 (información)
    '''
    simbolos = bits.reshape(-1,4)
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp*2)  # mpp: muestras por período
    portadora_I = np.cos(2*np.pi*fc*t_periodo)
    portadora_Q = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora_b1 = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
    moduladora_b2 = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
    moduladora_b3 = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
    moduladora_b4 = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
    
    # 4. Asignar las formas de onda según los bits (16-QAM)
    i = 0
    for i, simbolo in enumerate(simbolos):
        if simbolo[0] == 0 and simbolo[1] == 0:
            senal_Tx[i*mpp : (i+2)*mpp] = portadora_I *-3
            moduladora_b1[i*mpp : (i+2)*mpp] = 0
            moduladora_b2[i*mpp : (i+2)*mpp] = 0
        if simbolo[0] == 0 and simbolo[1] == 1:
            senal_Tx[i*mpp : (i+2)*mpp] = portadora_I *-1
            moduladora_b1[i*mpp : (i+2)*mpp] = 0
            moduladora_b2[i*mpp : (i+2)*mpp] = 1
        if simbolo[0] == 1 and simbolo[1] == 1:
            senal_Tx[i*mpp : (i+2)*mpp] = portadora_I *1
            moduladora_b1[i*mpp : (i+2)*mpp] = 1
            moduladora_b2[i*mpp : (i+2)*mpp] = 1
        if simbolo[0] == 1 and simbolo[1] == 0:
            senal_Tx[i*mpp : (i+2)*mpp] = portadora_I *3
            moduladora_b1[i*mpp : (i+2)*mpp] = 1
            moduladora_b2[i*mpp : (i+2)*mpp] = 0
        if simbolo[2] == 0 and simbolo[3] == 0:
            senal_Tx[(i+2)*mpp : (i+4)*mpp] = portadora_Q *3
            moduladora_b3[(i+2)*mpp : (i+4)*mpp] = 0
            moduladora_b4[(i+2)*mpp : (i+4)*mpp] = 0
        if simbolo[2] == 0 and simbolo[3] == 1:
            senal_Tx[(i+2)*mpp : (i+4)*mpp] = portadora_Q *1
            moduladora_b3[(i+2)*mpp : (i+4)*mpp] = 0
            moduladora_b4[(i+2)*mpp : (i+4)*mpp] = 1
        if simbolo[2] == 1 and simbolo[3] == 1:
            senal_Tx[(i+2)*mpp : (i+4)*mpp] = portadora_Q *-1
            moduladora_b3[(i+2)*mpp : (i+4)*mpp] = 1
            moduladora_b4[(i+2)*mpp : (i+4)*mpp] = 1
        if simbolo[2] == 1 and simbolo[3] == 0:
            senal_Tx[(i+2)*mpp : (i+4)*mpp] = portadora_Q *-3
            moduladora_b3[(i+2)*mpp : (i+4)*mpp] = 1
            moduladora_b4[(i+2)*mpp : (i+4)*mpp] = 0
    
    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, P_senal_Tx, portadora_I, portadora_Q, moduladora_b1, moduladora_b2, moduladora_b3, moduladora_b4

#Canal Ruidoso del 16-QAM:
def canal_ruidoso(senal_Tx, P_senal_Tx, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = P_senal_Tx / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido
    

    return senal_Rx

# Demodulación 16-QAM
def demodulador(senal_Rx, portadora_I, portadora_Q, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)
    
    # Cantidad de símbolos en transmisión
    N = int(M / mpp)
    Nbits = 4*N
    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(Nbits)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    Es_I = np.sum(portadora_I * portadora_I)
    Es_Q = np.sum(portadora_Q * portadora_Q)

    # Demodulación
    for i in range(0,N,4):
        # Producto interno de dos funciones
        producto_I = senal_Rx[i*mpp : (i+2)*mpp] * portadora_I
        producto_Q = senal_Rx[(i+2)*mpp : (i+4)*mpp] * portadora_Q
        Ep_I = np.sum(producto_I)
        Ep_Q = np.sum(producto_Q)
        senal_demodulada[i*mpp : (i+2)*mpp] = producto_I
        senal_demodulada[(i+2)*mpp : (i+4)*mpp] = producto_Q
        # Criterio de decisión por detección de energía para señal I
        if (Ep_I > 0*Es_I and Ep_I < 2*Es_I):
            bits_Rx[i] = 1
            bits_Rx[i+1] = 1
        if (Ep_I < 0*Es_I and Ep_I > -2*Es_I):
            bits_Rx[i] = 0
            bits_Rx[i+1] = 1
        if (Ep_I > 2*Es_I ):
            bits_Rx[i] = 1
            bits_Rx[i+1] = 0
        if (Ep_I < -2*Es_I ):
            bits_Rx[i] = 0
            bits_Rx[i+1] = 0
        

        # Criterio de decisión por detección de energía para señal Q
        if (Ep_Q > 0*Es_Q and Ep_Q < 2*Es_Q):
            bits_Rx[i+2] = 0
            bits_Rx[i+3] = 1
        if (Ep_Q < 0*Es_Q and Ep_Q > -2*Es_Q):
            bits_Rx[i+2] = 1
            bits_Rx[i+3] = 1
        if (Ep_Q > 2*Es_Q ):
            bits_Rx[i+2] = 0
            bits_Rx[i+3] = 0
        if (Ep_Q < -2*Es_Q ):
            bits_Rx[i+2] = 1
            bits_Rx[i+3] = 0
    

        

    return bits_Rx.astype(int), senal_demodulada

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)
    
    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)
    

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales,dimensiones)

    return pixeles.astype(np.uint8)

# Simulación de 16-QAM
import numpy as np
import matplotlib.pyplot as plt
import time


# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 5   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, P_senal_Tx, portadora_I, portadora_Q, moduladora_b1, moduladora_b2, moduladora_b3, moduladora_b4 = modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, P_senal_Tx, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora_I, portadora_Q, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

#Resolución del ejercicio 2 

# Se importa las librerías.
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Parámetros
fc = 5000  # frecuencia de ambas portadoras
p = 1 / 2

# Variables aleatorias A1 y A2.
va_A1 = stats.bernoulli(p)
va_A2 = stats.bernoulli(p)

# vector de tiempo
T = 100  # número de elementos
t_final = 10  # tiempo en segundos
t = np.linspace(0, t_final, T)

# Valor de  N 
N = 10000

# Funciones del tiempo S(t) con N realizaciones
S_t = np.empty((N, len(t)))

# Creación de las muestras del proceso s(t)
for i in range(N):
    A1 = va_A1.rvs()
    A2 = va_A2.rvs()

    if A1 == 0:
        A1 = -1
    else:
        A1 = 1

    if A2 == 0:
        A2 = -1
    else:
        A2 = 1

    s_t = A1 * np.cos(2 * np.pi * fc * t) + A2 * np.sin(2 * np.pi * fc * t)
    S_t[i, :] = s_t

    plt.plot(t, s_t)

# Promedio de s(t).
P = [np.mean(S_t[i, :]) for i in range(len(t))]
plt.plot(t, P, lw=4, label='Valor esperado teórico')

# Promedio de la senal_Tx.
P = [np.mean(senal_Tx) for i in range(len(t))]
plt.plot(t, P, '-.', lw=4, label='Valor esperado de la señal modulada')

# Mostrar las realizaciones, y su promedio calculado y teórico
plt.title('Realizaciones del proceso aleatorio $s(t)$')
plt.xlabel('$t$')
plt.ylabel('$s(t)$')
plt.legend()  # Se imprime las leyendas de la gráfica.
plt.show()  # Muestra la gráfica.

'''
Cáculo de la autocorrelación 
'''

# T valores de desplazamiento tau
desplazamiento = np.arange(T)
taus = desplazamiento / t_final

# Inicialización de matriz de valores de correlación para las N funciones
corr = np.empty((N, len(desplazamiento)))

# Nueva figura para la autocorrelación
plt.figure()

# Cálculo de correlación para cada valor de tau
for n in range(N):
    for i, tau in enumerate(desplazamiento):
        corr[n, i] = np.correlate(senal_Tx, np.roll(senal_Tx, tau)) / T
    plt.plot(taus, corr[n, :])

from scipy import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.show()

"""---

### Universidad de Costa Rica
#### Facultad de Ingeniería
##### Escuela de Ingeniería Eléctrica

---
"""