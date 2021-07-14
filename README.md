## Universidad de Costa Rica
### Escuela de Ingeniería Eléctrica
#### IE0405 - Modelos Probabilísticos de Señales y Sistemas
#### `P4` - *Modulación digital IQ*
---

* Estudiante: **Josué Vargas Matamoros**
* Carné: **B37321**
* Grupo: **2**
---
### 1. - Modulación 16-QAM

*  Realice una simulación del sistema de comunicaciones como en la sección 3.2., pero utilizando una modulación **16-QAM** en lugar de una modulación BPSK. Deben mostrarse las imágenes enviadas y recuperadas y las formas de onda.

En el archivo P4.py se puede encontrar el código implementado tanto en el modulador como en el demodulador para poder obtener el tipo de modulación 16-QAM. Se puede apreciar también que estas lógicas no contienen ningún error, mas no así la función bits_a_rgb, la cual arroja un error de tipo "ValueError". Investigando este error, se indica que este pasa cuando el vector a convertir de tamaño para este caso de 211464. Se quiere convertir a un vector de tamaño [89,198,3]. Cuando se realiza la operación 211464/89x198x3 da como resultado 4. El error se da generalmente cuando el resultado de esa división no es un un número entero, pero para este caso el resultado si es un número enter y no se entiende el error.

* Realice pruebas de estacionaridad y ergodicidad a la señal modulada `senal_Tx` y obtenga conclusiones sobre estas.

Las pruebas hechas con el código escrito para esta sección, nos da como resultado las siguientes imágenes:
