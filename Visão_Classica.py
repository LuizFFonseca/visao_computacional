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

    # ============================================================
    # MÉTODO ATUALIZADO COM MORFOLOGIA
    # ============================================================
    def predict(self, img, threshold):
        """Gera a máscara final e aplica morfologia para limpeza."""
        dist_map = self.get_distance_map(img)
        
        # 1. Gera máscara bruta (Distância menor que threshold = Corrosão)
        mask = (dist_map <= threshold).astype(np.uint8) * 255
        
        # 2. PASSO A: Reduzir Ruído (Opening)
        # Remove pixels isolados pequenos que não são corrosão real
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise)
        
        # 3. PASSO B: Unir Blobs (Closing ou Dilate)
        # Preenche buracos dentro da corrosão e junta áreas próximas
        # Um kernel maior aqui (ex: 7x7 ou 9x9) conecta partes mais distantes
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_merge)
        
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
        return melhor_th * 1.2

# ==========================================
# Métrica de Consistência
# ==========================================
def calculate_inside_ratio(mask_pred, yolo_boxes):
    total_pixels_detectados = np.count_nonzero(mask_pred)
    
    if total_pixels_detectados == 0:
        return 0.0 

    mask_yolo_combinada = np.zeros_like(mask_pred)

    for (x1, y1, x2, y2) in yolo_boxes:
        cv2.rectangle(mask_yolo_combinada, (x1, y1), (x2, y2), 255, -1)

    intersecao = cv2.bitwise_and(mask_pred, mask_yolo_combinada)
    pixels_dentro = np.count_nonzero(intersecao)

    ratio = pixels_dentro / total_pixels_detectados
    return ratio

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    
    # 1. Definição de Pastas
    pasta_teste_raiz = "corrosion detect/images"
    pasta_labels_teste = "corrosion detect/labels" 
    
    # Busca os arquivos dentro das pastas
    path_imgs_treino = sorted(glob.glob("corrosion detect/images_train/*.jpeg"))
    path_masks_treino = sorted(glob.glob("corrosion detect/mask/*.png"))
    
    # Busca imagens de teste
    path_imgs_teste = []
    for ext in ['*.jpeg', '*.jpg', '*.png']:
        path_imgs_teste.extend(glob.glob(os.path.join(pasta_teste_raiz, ext)))
    path_imgs_teste.sort()

    if not path_imgs_treino:
        print("ERRO: Nenhuma imagem de treino encontrada.")
        exit()
    
    if not os.path.exists(pasta_labels_teste):
        print(f"AVISO: Pasta de labels '{pasta_labels_teste}' não encontrada.")

    # 2. Instanciar e Treinar
    detector = CorrosionDetector()
    detector.fit(path_imgs_treino, path_masks_treino)

    # 3. Otimizar Threshold
    best_th = detector.optimize_threshold_on_train_set(path_imgs_treino, path_masks_treino)

    # 4. Visualização com Métrica
    print("-" * 30)
    print(f"Iniciando visualização e cálculo da métrica")
    print("-" * 30)

    for teste_path in path_imgs_teste:
        print(f"Processando: {teste_path}")
        img_teste = cv2.imread(teste_path)
        
        if img_teste is None: continue

        # Predição (AGORA COM MORFOLOGIA)
        mask_resultado, dist_map = detector.predict(img_teste, threshold=best_th)
        
        # --- PREPARAÇÃO VISUAL ---
        img_rgb = cv2.cvtColor(img_teste, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        overlay[mask_resultado > 0] = [255, 0, 0] # Segmentação em Vermelho
        img_final_overlay = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

        # --- PROCESSAMENTO YOLO (Acumular Boxes) ---
        nome_arquivo = os.path.basename(teste_path)
        nome_txt = os.path.splitext(nome_arquivo)[0] + ".txt"
        caminho_txt = os.path.join(pasta_labels_teste, nome_txt)
        
        lista_boxes_validas = [] 
        tem_label = False

        if os.path.exists(caminho_txt):
            h_img, w_img = img_teste.shape[:2]
            tem_label = True
            
            with open(caminho_txt, 'r') as f:
                linhas = f.readlines()
            
            for linha in linhas:
                dados = linha.strip().split()
                if len(dados) >= 5:
                    x_c_norm, y_c_norm = float(dados[1]), float(dados[2])
                    w_norm, h_norm = float(dados[3]), float(dados[4])

                    w_box, h_box = int(w_norm * w_img), int(h_norm * h_img)
                    x_center, y_center = int(x_c_norm * w_img), int(y_c_norm * h_img)

                    x1 = int(x_center - w_box / 2)
                    y1 = int(y_center - h_box / 2)
                    x2 = int(x_center + w_box / 2)
                    y2 = int(y_center + h_box / 2)
                    
                    lista_boxes_validas.append((x1, y1, x2, y2))
                    cv2.rectangle(img_final_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # --- CÁLCULO DA MÉTRICA ---
        metric_score = 0.0
        msg_metrica = "Sem Label YOLO"
        
        if tem_label:
            metric_score = calculate_inside_ratio(mask_resultado, lista_boxes_validas)
            msg_metrica = f"Inside: {metric_score*100:.1f}%"
            
            cv2.putText(img_final_overlay, msg_metrica, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            print(f"   -> {msg_metrica} dos pixels previstos estão dentro da caixa.")

        # --- VISUALIZAÇÃO ---
        plt.figure(figsize=(16, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Imagem Original")
        plt.imshow(img_rgb)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title("Distância Mahalanobis")
        plt.imshow(dist_map, cmap='jet')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title(f"Segmentação + Morfologia")
        plt.imshow(mask_resultado, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title(f"Segm (Verm) + YOLO (Verde)\n{msg_metrica}")
        plt.imshow(img_final_overlay)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()