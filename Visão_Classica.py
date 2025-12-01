import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

class CorrosionDetector:
    def __init__(self):
        self.mean = None
        self.inv_cov = None
        self.trained = False

    def fit(self, image_paths, mask_paths):
        """Treina o modelo (aprende a cor da corrosão)."""
        print(f"[TREINO] Carregando {len(image_paths)} imagens de treino...")
        all_samples = []

        for img_path, mask_path in zip(image_paths, mask_paths):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None: continue
            
            # Garante binário
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Extrai pixels de corrosão (Espaço LAB)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            pixels = img_lab[mask > 0]
            
            if len(pixels) > 0:
                all_samples.append(pixels)

        if not all_samples:
            raise ValueError("Erro: Nenhum pixel de corrosão encontrado nas máscaras.")

        training_data = np.vstack(all_samples).astype(np.float32)
        
        # Estatísticas (Mahalanobis)
        self.mean = np.mean(training_data, axis=0)
        cov = np.cov(training_data, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6 # Regularização
        self.inv_cov = np.linalg.inv(cov)
        self.trained = True
        print("[TREINO] Concluído.")

    def get_distance_map(self, img):
        """Retorna o mapa de calor das distâncias."""
        if not self.trained: raise RuntimeError("Modelo não treinado.")
        
        h, w = img.shape[:2]
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        img_flat = img_lab.reshape(-1, 3)
        
        delta = img_flat - self.mean
        # D^2 = (x-u)T * S^-1 * (x-u)
        temp = np.dot(delta, self.inv_cov)
        dist_sq = np.sum(temp * delta, axis=1)
        
        return np.sqrt(dist_sq).reshape(h, w)

    def predict(self, img, threshold):
        """Gera a máscara final e retorna também o mapa de distância."""
        dist_map = self.get_distance_map(img)
        
        # Gera máscara (Distância menor que threshold = Corrosão)
        mask = (dist_map <= threshold).astype(np.uint8) * 255
        
        # Limpeza básica (opcional)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, dist_map

    def optimize_threshold_on_train_set(self, image_paths, mask_paths):
        """Encontra o melhor threshold testando no próprio treino."""
        print(f"[OTIMIZAÇÃO] Buscando melhor threshold...")
        
        # 1. Pré-calcular mapas para agilidade
        all_dists = []
        all_labels = []

        for img_path, mask_path in zip(image_paths, mask_paths):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask.shape != img.shape[:2]:
                 mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            dist_map = self.get_distance_map(img)
            
            all_dists.append(dist_map.flatten())
            all_labels.append(mask_bin.flatten())

        y_dists = np.concatenate(all_dists)
        y_true = np.concatenate(all_labels)

        # 2. Testar valores
        thresholds = np.arange(1.0, 10.0, 0.1)
        melhor_f1 = 0
        melhor_th = 3.0

        for th in thresholds:
            y_pred = (y_dists <= th).astype(np.uint8)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            denom = (2 * tp + fp + fn)
            f1 = (2 * tp) / denom if denom > 0 else 0

            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_th = th

        print(f"--> Threshold Ideal Encontrado: {melhor_th:.1f} (F1: {melhor_f1:.4f})")
        return melhor_th*1.2

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    
    # 1. Definição de Pastas (Conforme seu pedido)
    pasta_teste_raiz = "corrosion detect/images"
    
    # Busca os arquivos dentro das pastas
    path_imgs_treino = sorted(glob.glob("corrosion detect/images_train/*.jpeg"))
    path_masks_treino = sorted(glob.glob("corrosion detect/mask/*.png"))
    
    # Para teste, vamos pegar todos os formatos comuns dentro da pasta indicada
    path_imgs_teste = []
    for ext in ['*.jpeg', '*.jpg', '*.png']:
        path_imgs_teste.extend(glob.glob(os.path.join(pasta_teste_raiz, ext)))
    path_imgs_teste.sort()

    # Validações básicas
    if not path_imgs_treino:
        print("ERRO: Nenhuma imagem de treino encontrada em 'corrosion detect/images_train/*.jpeg'")
        exit()
    if not path_imgs_teste:
        print(f"AVISO: Nenhuma imagem de teste encontrada em '{pasta_teste_raiz}'")

    # 2. Instanciar e Treinar
    detector = CorrosionDetector()
    detector.fit(path_imgs_treino, path_masks_treino)

    # 3. Otimizar Threshold (Usando o Treino)
    best_th = detector.optimize_threshold_on_train_set(path_imgs_treino, path_masks_treino)

    # 4. Testar em novas imagens com OVERLAY (Visualização Melhorada)
    print("-" * 30)
    print(f"Iniciando visualização com Threshold Otimizado: {best_th}")
    print("-" * 30)

    for teste_path in path_imgs_teste:
        print(f"Processando imagem de teste: {teste_path}")
        img_teste = cv2.imread(teste_path)
        
        if img_teste is None:
            print(f"Erro ao abrir {teste_path}")
            continue

        # Predição
        mask_resultado, dist_map = detector.predict(img_teste, threshold=best_th)
        
        # --- CRIANDO O OVERLAY (MÁSCARA VERMELHA TRANSLÚCIDA) ---
        # 1. Converter imagem original para RGB (Matplotlib usa RGB, OpenCV usa BGR)
        img_rgb = cv2.cvtColor(img_teste, cv2.COLOR_BGR2RGB)
        
        # 2. Criar uma cópia para pintar de vermelho sólido onde tem corrosão
        overlay = img_rgb.copy()
        
        # 3. Definir a cor da máscara (Vermelho: R=255, G=0, B=0)
        cor_mask = [255, 0, 0] 
        
        # 4. Pintar a região da máscara na cópia 'overlay'
        overlay[mask_resultado > 0] = cor_mask
        
        # 5. Misturar a imagem original com a pintada (Alpha Blending)
        # alpha=0.6 (60% original), beta=0.4 (40% vermelho)
        img_final_overlay = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

        # --- VISUALIZAÇÃO ---
        plt.figure(figsize=(16, 5)) # Aumentei a largura da figura
        
        # Plot 1: Original
        plt.subplot(1, 4, 1)
        plt.title(f"Imagem Original ({os.path.split(teste_path)[-1]})")
        plt.imshow(img_rgb)
        plt.axis('off')
        
        # Plot 2: Mapa de Calor (Explicação Matemática)
        plt.subplot(1, 4, 2)
        plt.title("Distância Mahalanobis")
        plt.imshow(dist_map, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04) # Ajuste visual da barra
        plt.axis('off')
        
        # Plot 3: Máscara Binária (O que o computador 'vê')
        plt.subplot(1, 4, 3)
        plt.title(f"Segmentação P&B (Th={best_th:.1f})")
        plt.imshow(mask_resultado, cmap='gray')
        plt.axis('off')
        
        # Plot 4: OVERLAY (Para humanos visualizarem)
        plt.subplot(1, 4, 4)
        plt.title("Visualização Final")
        plt.imshow(img_final_overlay)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()