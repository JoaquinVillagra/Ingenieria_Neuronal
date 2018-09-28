# FeedFoward : Propagacion hacia adelante
def prediction(x, w, bias=1):
    i = 1
    aux = activacion_sigmoide(x.dot(w[i-1]) + bias)
    for i in range(len(w)):
        aux = activacion_sigmoide(c1.dot(w[i]) + bias)

    return aux

#Entrenamiento de la red neuronal
def entrenamiento(entradas, salidas, neuronas, iteraciones, tasa_aprendizaje):
    #Asignación de pesos
    w1 = np.random.rand(entradas.shape[1], neuronas)
    w2 = np.random.rand(neuronas, salidas.shape[1])
    w  = [w1, w2]
    errores = []
    for iteracion in range(iteraciones):
        for entrada, salida in zip(entradas, salidas):
            #forward
            l0 = entrada
            l1 = activacion_sigmoide(np.dot(l0, w1) + bias)
            l2 = activacion_sigmoide(np.dot(l1, w2) + bias)
            
            #Aqui comienza el proceso de aprendizaje - BackPropagation
            l2_error = -2*(l2-salida)
            
            #calculo de deltas
            l2_delta = np.multiply(l2_error, activacion_sigmoide_prima(l2.A1))
            l1_error = np.dot(l2_delta, w2.T)
            l1_delta = np.multiply(l1_error, activacion_sigmoide_prima(l1.A1))
            
            #Actualización de Pesos
            w2 = w2 + tasa_aprendizaje*np.dot(l1.T, l2_delta)
            w1 = w1 + tasa_aprendizaje*np.dot(l0.T, l1_delta)
            w  = [w1, w2]
            
        #Probando - Calculando error
        errores.append(np.square(np.subtract(prediction(entradas, w), salidas)).mean(axis=0).A1)
        if(np.square(np.subtract(prediction(entradas, w), salidas)).mean(axis=0).A1[0] < 0.01):
            break

    return w1, w2, errores