import numpy as np

class Activation_Fonctions:
    
    def ReLU_activation(self, x, derivate=False):
        """
        Fonction d'activation ReLU (Rectified Linear Unit).
        
        Input:
            x (np.ndarray): Tableau numpy contenant les valeurs d'entrée
            derivate (bool, optional): Si True, retourne la dérivée de ReLU. Defaults to False.
        
        Output:
            np.ndarray: 
                - Si derivate=False: max(0, x) pour chaque élément
                - Si derivate=True: 1 si x > 0, 0 sinon
        """
        if derivate:
            return (x > 0).astype(float)
        else:
            return np.maximum(0, x)

    def sigmoid_activation(self, x, derivate=False):
        """
        Fonction d'activation sigmoïde.
        
        Input:
            x (np.ndarray): Tableau numpy contenant les valeurs d'entrée
            derivate (bool, optional): Si True, retourne la dérivée de la sigmoïde. Defaults to False.
        
        Output:
            np.ndarray: 
                - Si derivate=False: 1 / (1 + exp(-x)) pour chaque élément
                - Si derivate=True: sigmoid(x) * (1 - sigmoid(x))
        """
        sigmoid_value = 1 / (1 + np.exp(-x))
        if derivate:
            return sigmoid_value * (1 - sigmoid_value)
        else:
            return sigmoid_value