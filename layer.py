import numpy as np

class Layer :
    
    def __init__(self, number_of_inputs, number_of_neurons, activation_fonction) :
        """
        Initialise une couche de neurones.
        
        Input:
            number_of_inputs (int): Nombre de neurones dans la couche précédente
            number_of_neurons (int): Nombre de neurones dans cette couche
            activation_fonction (callable): Fonction d'activation à utiliser
        
        Output:
            None: Initialise les attributs de la couche (weights, biais, etc.)
        """
        self.weights = np.random.randn(number_of_inputs, number_of_neurons) * np.sqrt(2/number_of_inputs)
        self.biais = np.zeros((1, number_of_neurons))
        self.activation_fonction = activation_fonction

        #Pour la backpropagation
        self.inputs = None
        self.z = None
        self.output = None  

        #Pour les calculs de gradient
        self.dW = None
        self.db = None

    def forward(self, inputs) :
        """
        Propagation avant (forward pass) de la couche.
        
        Input:
            inputs (np.ndarray): Tableau numpy de shape (batch_size, number_of_inputs) contenant les entrées
        
        Output:
            np.ndarray: Tableau numpy de shape (batch_size, number_of_neurons) contenant les sorties activées
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biais
        return self.activation_fonction(self.z)

    def backward(self, dL_da, learning_rate) :
        """
        Propagation arrière (backward pass) de la couche.
        
        Input:
            dL_da (np.ndarray): Gradient de la perte par rapport à la sortie activée de cette couche
            learning_rate (float): Taux d'apprentissage pour la mise à jour des poids
        
        Output:
            np.ndarray: Gradient de la perte par rapport à la sortie de la couche précédente (dL_da_prev)
        """
        batch_size = self.inputs.shape[0]

        #Gradient δL/δz = δL/da * δa/δz
        dL_dz = dL_da * self.activation_fonction(self.z, True)

        #Gradient δL/δW = 1/batch_size * Transposée(inputs) * δL/δz
        self.dW = (1/batch_size) * np.dot(self.inputs.T, dL_dz)

        #Gradient δL/δb = 1/batch_size * sum(δL/δz, axis 0)
        self.db = (1/batch_size) * np.sum(dL_dz, axis=0, keepdims=True)

        #Gradient δL/δa_prev = δL/δz * Transposée(W)
        dL_da_prev = np.dot(dL_dz, self.weights.T)

        #Evolution des poids et des biais
        self.weights -= learning_rate * self.dW
        self.biais -= learning_rate * self.db
        
        return dL_da_prev