import numpy as np
import csv


# Charge CSV 
def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.reader(f) 
        headers = next(reader)  
        rows = []   
        for row in reader:
            rows.append(row) 
    return headers, np.array(rows)




# Extraire data
def prepare_data(train_headers, train_data, test_headers, test_data):
    id_col = test_headers[0]       
    label_col= train_headers[-1]     
    features  = test_headers[1:8]      


    test_ids = test_data[:,test_headers.index(id_col)]
    X_train= train_data[:, [train_headers.index(f) for f in features]].astype(float)
    y_train = train_data[:, train_headers.index(label_col)]
    X_test  = test_data[:,[test_headers.index(f)  for f in features]].astype(float)


    return X_train, y_train, X_test, test_ids, id_col, label_col, features




# Supprimer colonne
def drop_columns(X_train, X_test, feature_names, cols_to_drop):
    
    keep = [i for i, name in enumerate(feature_names) if name not in cols_to_drop]
    return X_train[:, keep], X_test[:, keep]


# Normalisation Min-Max 
def minmax_normalize(X_train, X_test):
    col_min = X_train.min(axis=0)
    col_max = X_train.max(axis=0)
    spread  =col_max - col_min
    spread[spread==0] = 1
    X_train_norm =(X_train - col_min) / spread
    X_test_norm  = (X_test - col_min) / spread
    return X_train_norm, X_test_norm



#  Train (90%) test (10%)
def train_val_split(X, y, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    shuffled = np.random.permutation(len(X))
    cut= int(len(X) * (1 - val_ratio))
    train_idx = shuffled[:cut]
    
    val_idx = shuffled[cut:]
    
    
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]





# knn
def knn_predict(X_train, y_train, X_test, k):
    distances = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))

    predictions = []
    for i in range(len(X_test)):
        d = distances[i]
        neighbors = np.argsort(d)[:k]          
        labels = y_train[neighbors]   
        weights = 1.0 / (d[neighbors] + 1e-10)
        class_scores = {}
        
        for label, w in zip(labels, weights):
            class_scores[label] = class_scores.get(label, 0) + w
        predictions.append(max(class_scores, key=class_scores.get))  

    return np.array(predictions)





# Précision 
def cross_validate(X, y, k, n_folds=20):
    np.random.seed(42) 
    indices = np.random.permutation(len(X)) 
    folds   = np.array_split(indices, n_folds)   

    fold_scores = []
    for i in range(n_folds):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

        preds = knn_predict(X[train_idx], y[train_idx], X[val_idx], k)
        fold_scores.append(np.mean(preds == y[val_idx])) 


    return np.mean(fold_scores), np.std(fold_scores)


# Tester K 1-20.
def find_best_k(X_train, y_train):
    print(f"  {'K':>3}  {'Précision':>10}  {'Écart-type':>11}  Stabilité")
    print("  " + "-" * 42)

    best_k, best_score, best_std = 1, 0.0, 1.0 


    for k in range(1, 21):
        score, std = cross_validate(X_train, y_train, k) 
        stable = "s" if std < 0.015 else ("ms" if std < 0.03 else "ps") 
        
        
        print(f"  {k:>3}  {score*100:>9.3f}%  {std*100:>10.3f}%  {stable}")

        if (score - 0.5 * std) > (best_score - 0.5 * best_std): 
            best_k, best_score, best_std = k, score, std 

    print(f"\n  → Meilleur K = {best_k}  ({best_score*100:.3f}% ± {best_std*100:.3f}%)\n")
    return best_k, best_score, best_std


# Tester 50 seeds
def find_best_split(X, y, k):
    print("  Test de 50 découpages aléatoires...")
    best_seed, best_score = 0, 0.0

    for seed in range(50):
        X_tr, X_val, y_tr, y_val = train_val_split(X, y, seed=seed)   
        X_tr_n, X_val_n= minmax_normalize(X_tr, X_val)
        preds= knn_predict(X_tr_n, y_tr, X_val_n, k)
        score = np.mean(preds == y_val)   

        if score > best_score:
            best_score, best_seed = score, seed

    print(f"  → Meilleur découpage : seed={best_seed}  ({best_score*100:.3f}%)\n")
    return best_seed


# Affichage
def print_results(y_true, y_pred, classes, k, cv_score, cv_std, best_seed):
    # Matrice de confusion
    idx    = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)   
    for true, pred in zip(y_true, y_pred):   
        matrix[idx[true]][idx[pred]] += 1 

    accuracy = np.mean(y_true == y_pred)   

    print("\n" + "=" * 60)
    print("  RÉSULTATS")
    print("=" * 60)
    print(f"  Colonne supprimée   : C5 (bruit)")
    print(f"  Normalisation       : Min-Max [0, 1]")     
    print(f"  Distance            : Euclidean")
    print(f"  K                   : {k}")
    print(f"  Seed                : {best_seed}")  
    print(f"  CV 20-fold          : {cv_score*100:.3f}% ± {cv_std*100:.3f}%") 
    print(f"  Validation          : {accuracy*100:.3f}%")


    print("\n  MATRICE DE CONFUSION")   
    print("  " + "-" * (len(classes) * 10 + 12))
    print("  Réel \\ Prédit  " + "  ".join(f"{c:>6}" for c in classes))
    print("  " + "-" * (len(classes) * 10 + 12)) 
    for i, c in enumerate(classes):
        print("  " + f"{c:>14}  " + "  ".join(f"{matrix[i,j]:>6}" for j in range(len(classes))))
    print("  " + "-" * (len(classes) * 10 + 12))
  
    print("\n  MÉTRIQUES PAR CLASSE")
    print(f"  {'Classe':<12} {'Précision':>10} {'Rappel':>10} {'F1':>10}")  
    print("  " + "-" * 44)
    precisions, recalls, f1s = [], [], []
    
    
    for i, c in enumerate(classes):  
        tp = matrix[i, i] 
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp  
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0   
        precisions.append(p); recalls.append(r); f1s.append(f1)
        print(f"  {c:<12} {p*100:>9.2f}% {r*100:>9.2f}% {f1*100:>9.2f}%")
    print("  " + "-" * 44)
    print(f"  {'Moyenne':<12} {np.mean(precisions)*100:>9.2f}% "    
          f"{np.mean(recalls)*100:>9.2f}% {np.mean(f1s)*100:>9.2f}%")
    print("=" * 60 + "\n")





# Sauvegarde CSV   
def save_results(path, test_ids, predictions, id_col, label_col):
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([id_col, label_col])
        for id_, pred in zip(test_ids, predictions):
            writer.writerow([id_, pred])


#  Main 
if __name__ == "__main__":

    # 1. Chargement des données
    train_headers, train_data = load_csv("train.csv")
    test_headers,  test_data = load_csv("test.csv")   
   
    X_train, y_train, X_test, test_ids, id_col, label_col, features = prepare_data(
        train_headers, train_data, test_headers, test_data)
   
    # 2. Suppression de C5 
    X_train, X_test = drop_columns(X_train, X_test, features, ["C5"])



    # 3. Découpage train test et normalsiation  
    X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train, seed=42)
    X_tr_n, X_val_n = minmax_normalize(X_tr, X_val)

    print("recherche du meilleur k\n")
    best_k, best_cv, best_std = find_best_k(X_tr_n, y_tr)


    # 4. Recherche du meilleur découpage 
    print("Recherche du meilleur découpage\n") 
    best_seed = find_best_split(X_train, y_train, best_k) 

    # 5. Test finale sur le meilleur découpage
    X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train, seed=best_seed)  
    X_tr_n, X_val_n= minmax_normalize(X_tr, X_val)
    val_preds= knn_predict(X_tr_n, y_tr, X_val_n, best_k) 
    classes = np.unique(y_train) 
    print_results(y_val, val_preds, classes, best_k, best_cv, best_std, best_seed)

    # 6. Entraînement sur 100% des données + prédiction test 
    X_train_n, X_test_n = minmax_normalize(X_train, X_test)   
    final_predictions = knn_predict(X_train_n, y_train, X_test_n, best_k)   

    # 7. Enregistrement 
    save_results("submission.csv", test_ids, final_predictions, id_col, label_col)