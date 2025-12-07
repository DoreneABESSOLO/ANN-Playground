import numpy as np
from layer import Layer

class ANN:

    def __init__(self, learning_rate, size_of_layers, initial_input_size, activation_fonction):
        """
        Initialise un réseau de neurones multicouche (MLP).
        
        Input:
            learning_rate (float): Taux d'apprentissage pour l'optimisation
            size_of_layers (list): Liste contenant le nombre de neurones pour chaque couche cachée
            initial_input_size (int): Nombre de caractéristiques d'entrée
            activation_fonction (callable): Fonction d'activation à utiliser pour toutes les couches
        
        Output:
            None: Initialise le réseau avec les couches spécifiées + une couche de sortie à 3 neurones
        """
        self.learning_rate = learning_rate
        self.layers = []
        for i in range(len(size_of_layers)):
            if i == 0:
                self.add(initial_input_size, size_of_layers[i], activation_fonction)
            else:
                self.add(size_of_layers[i-1], size_of_layers[i], activation_fonction)
        self.add(size_of_layers[-1], 3, activation_fonction)

    def add(self, number_of_inputs, number_of_neurons, activation_fonction):
        """
        Ajoute une nouvelle couche au réseau.
        
        Input:
            number_of_inputs (int): Nombre de neurones dans la couche précédente
            number_of_neurons (int): Nombre de neurones dans la nouvelle couche
            activation_fonction (callable): Fonction d'activation pour cette couche
        
        Output:
            None: Ajoute une couche à la liste self.layers
        """
        self.layers.append(Layer(number_of_inputs ,number_of_neurons, activation_fonction))

    def forward(self, inputs):
        """
        Propagation avant à travers toutes les couches du réseau.
        
        Input:
            inputs (np.ndarray): Tableau numpy de shape (batch_size, initial_input_size) contenant les données d'entrée
        
        Output:
            np.ndarray: Tableau numpy de shape (batch_size, 3) contenant les probabilités après softmax
        """
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return self.softmax(current_input)

    def softmax(self, inputs):
        """
        Applique la fonction softmax pour obtenir des probabilités.
        
        Input:
            inputs (np.ndarray): Tableau numpy de shape (batch_size, num_classes) contenant les logits
        
        Output:
            np.ndarray: Tableau numpy de même shape contenant les probabilités normalisées (somme = 1 par ligne)
        """
        exp_inp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_inp / np.sum(exp_inp, axis=1, keepdims=1)

    def backward(self, y_true, y_pred):
        """
        Propagation arrière (backpropagation) à travers toutes les couches.
        
        Input:
            y_true (np.ndarray): Tableau numpy de shape (batch_size, 3) contenant les labels réels (one-hot)
            y_pred (np.ndarray): Tableau numpy de shape (batch_size, 3) contenant les prédictions
        
        Output:
            None: Met à jour les poids et biais de toutes les couches via gradient descent
        """
        #Gradient de la dernière couche
        dL_da = y_pred - y_true
        for layer in reversed(self.layers):
            dL_da = layer.backward(dL_da, self.learning_rate)

    def cross_entropy(self, y_true, y_pred):
        """
        Calcule la perte d'entropie croisée (cross-entropy loss).
        
        Input:
            y_true (np.ndarray): Tableau numpy de shape (batch_size, num_classes) contenant les labels réels (one-hot)
            y_pred (np.ndarray): Tableau numpy de shape (batch_size, num_classes) contenant les prédictions
        
        Output:
            float: Valeur de la perte moyenne sur le batch
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Loss moyenne sur le batch
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def fit(self, inputs, y_true, batch_size=None, validation_data=None, verbose=True, patience=50, min_delta=1e-6):
        """
        Entraîne le réseau de neurones sur les données fournies.
        
        Input:
            inputs (np.ndarray): Tableau numpy de shape (n_samples, initial_input_size) contenant les données d'entrée
            y_true (np.ndarray): Tableau numpy de shape (n_samples, 3) contenant les labels réels (one-hot)
            batch_size (int, optional): Taille des batches. Si None, utilise tout le dataset. Defaults to None.
            validation_data (tuple, optional): Tuple (X_val, y_val) pour calculer la loss de validation. Defaults to None.
            verbose (bool, optional): Affiche les logs d'entraînement. Defaults to True.
            patience (int, optional): Nombre d'epochs à attendre sans amélioration avant d'arrêter. Defaults to 50.
            min_delta (float, optional): Amélioration minimale pour considérer une amélioration. Defaults to 1e-6.
        
        Output:
            dict: Historique de l'entraînement avec 'train_loss', 'val_loss', 'best_epoch', 'best_val_loss'
        """
        max_epoch = 10000
        epoch = 0
        
        history = {'train_loss': [], 'val_loss': [], 'best_epoch': 0, 'best_val_loss': np.inf, 
                   'stopped_epoch': 0, 'stop_reason': 'unknown'}
        
        # Early stopping avec patience
        best_val_loss = np.inf
        patience_counter = 0
        best_epoch = 0
        
        # Si batch_size n'est pas spécifié, utiliser tout le dataset
        if batch_size is None:
            batch_size = len(inputs)

        while epoch < max_epoch:
            # Mélanger les données à chaque epoch
            indices = np.random.permutation(len(inputs))
            inputs_shuffled = inputs[indices]
            y_true_shuffled = y_true[indices]
            
            # Diviser en batches
            n_batches = int(np.ceil(len(inputs) / batch_size))
            epoch_losses = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(inputs))
                
                batch_inputs = inputs_shuffled[start_idx:end_idx]
                batch_y_true = y_true_shuffled[start_idx:end_idx]
                
                # Forward pass
                batch_y_pred = self.predict(batch_inputs)
                batch_loss = self.cross_entropy(batch_y_true, batch_y_pred)
                epoch_losses.append(batch_loss)
                
                # Backward pass
                self.backward(batch_y_true, batch_y_pred)
            
            # Loss moyenne sur tous les batches de l'epoch
            avg_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_loss)
            
            # Calculer la loss de validation si fournie
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                val_loss = self.cross_entropy(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
                
                # Early stopping basé sur la validation loss
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    history['best_epoch'] = best_epoch
                    history['best_val_loss'] = best_val_loss
                else:
                    patience_counter += 1
                
                if verbose:
                    print(f"Epoch {epoch}, train_loss={avg_loss:.5f}, val_loss={val_loss:.5f}, best_val_loss={best_val_loss:.5f} (epoch {best_epoch})")
                
                # Arrêter si patience atteinte
                if patience_counter >= patience:
                    history['stopped_epoch'] = epoch
                    history['stop_reason'] = f'Early stopping (patience={patience}, meilleur à epoch {best_epoch})'
                    if verbose:
                        print(f"Early stopping à l'epoch {epoch} (pas d'amélioration depuis {patience} epochs, meilleur à l'epoch {best_epoch})")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch}, loss={avg_loss:.5f}")

            epoch += 1
        
        # Si on a atteint max_epoch sans early stopping
        if epoch >= max_epoch and history['stop_reason'] == 'unknown':
            history['stopped_epoch'] = epoch - 1
            history['stop_reason'] = f'Max epochs atteint ({max_epoch})'
        elif history['stop_reason'] == 'unknown':
            history['stopped_epoch'] = epoch - 1
            history['stop_reason'] = 'Entraînement terminé normalement'

        return history

    def predict(self, inputs):
        """
        Effectue une prédiction sur les données d'entrée.
        
        Input:
            inputs (np.ndarray): Tableau numpy de shape (batch_size, initial_input_size) contenant les données d'entrée
        
        Output:
            np.ndarray: Tableau numpy de shape (batch_size, 3) contenant les probabilités de prédiction
        """
        return self.forward(inputs)