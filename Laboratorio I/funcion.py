def aprendizaje(tasa, entradas, salidas_deseadas pesos, iteraciones):
    '''
        tasa     = tasa de aprendizaje de la red neuronal [Valor entre 0 y 1]
        entradas = vector de entradas a procesar
        salidas_deseadas = vector de salidas deseadas de la red
        pesos    = vector de pesos que multiplicaran a las entradas de la red 
    '''
    errores = []
    for i in range(iteraciones):
        cantidad_entradas = len(entradas)
        for k in range(cantidad_entradas)
            salidas = activacion(entradas[k].dot(pesos))
            error = salidas_deseadas[k] - salidas
            errores.append(int(error))
            pesos += (tazaAprendizaje*(error)*entradas[k]).T

    return errores
       