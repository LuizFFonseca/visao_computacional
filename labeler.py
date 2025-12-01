import cv2
import numpy as np
import os
import glob

# --- CONFIGURAÇÕES ---
PASTA_IMAGENS = "corrosion detect/images_train" 
PASTA_MASKS = "corrosion detect/mask"
COR_PINCEL = 255       
COR_BORRACHA = 0       
COR_VISUALIZACAO = (0, 0, 255) # Vermelho

class ImageLabelerZoom:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        self.image_paths.sort()
        
        if not self.image_paths:
            print(f"Nenhuma imagem encontrada em: {input_dir}")
            exit()

        self.current_idx = 0
        self.brush_size = 5 # Começa menor pois o zoom ajuda
        self.drawing = False
        self.erasing = False
        self.mask = None
        self.img = None
        self.window_name = "Labeler com Zoom (Role o mouse)"

        # Variáveis de Zoom e Pan
        self.zoom_level = 1.0 # 1.0 = 100%
        self.pan_x = 0 # Offset X da visualização na imagem original
        self.pan_y = 0 # Offset Y da visualização na imagem original

    def get_real_coords(self, x_win, y_win):
        """Converte coordenada da tela para coordenada real da imagem."""
        # A lógica é: (CoordTela / Zoom) + DeslocamentoPan
        real_x = int(x_win / self.zoom_level) + self.pan_x
        real_y = int(y_win / self.zoom_level) + self.pan_y
        
        # Clamp para garantir que não saia da imagem
        h, w = self.img.shape[:2]
        real_x = max(0, min(real_x, w - 1))
        real_y = max(0, min(real_y, h - 1))
        
        return real_x, real_y

    def apply_brush(self, x, y, color):
        """Aplica o pincel na máscara real."""
        cv2.circle(self.mask, (x, y), self.brush_size, color, -1)

    def mouse_callback(self, event, x, y, flags, param):
        # Mapeia coordenadas da tela para a imagem real
        rx, ry = self.get_real_coords(x, y)

        # Scroll do Mouse (Zoom)
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0: # Scroll Up
                self.zoom_level += 0.1
            else: # Scroll Down
                self.zoom_level = max(0.5, self.zoom_level - 0.1) # Minimo 50%
            self.update_pan_limits() # Ajusta para não sair da tela ao dar zoom out

        # Botão Esquerdo (Pintar)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.apply_brush(rx, ry, COR_PINCEL)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.apply_brush(rx, ry, COR_PINCEL)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

        # Botão Direito (Apagar)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
            self.apply_brush(rx, ry, COR_BORRACHA)
        elif event == cv2.EVENT_MOUSEMOVE and self.erasing:
            self.apply_brush(rx, ry, COR_BORRACHA)
        elif event == cv2.EVENT_RBUTTONUP:
            self.erasing = False

    def update_pan_limits(self):
        """Garante que a janela de visualização não saia dos limites da imagem."""
        h, w = self.img.shape[:2]
        
        # Tamanho da janela visível em pixels reais
        visible_w = int(w / self.zoom_level)
        visible_h = int(h / self.zoom_level)
        
        # O Pan máximo é o tamanho da imagem menos o que estamos vendo
        max_pan_x = max(0, w - visible_w)
        max_pan_y = max(0, h - visible_h)
        
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))

    def get_view_image(self):
        """Gera a imagem que será mostrada na tela (Crop + Resize)."""
        h, w = self.img.shape[:2]
        
        # 1. Cria a visualização combinada (Imagem + Máscara Vermelha)
        display_full = self.img.copy()
        mask_indices = self.mask > 0
        if np.any(mask_indices):
            overlay = display_full.copy()
            overlay[mask_indices] = COR_VISUALIZACAO
            cv2.addWeighted(overlay, 0.4, display_full, 0.6, 0, display_full)

        # 2. Cortar a região de interesse (ROI) baseada no Pan e Zoom
        visible_w = int(w / self.zoom_level)
        visible_h = int(h / self.zoom_level)
        
        # Ajuste fino para não estourar array slice
        end_x = min(w, self.pan_x + visible_w)
        end_y = min(h, self.pan_y + visible_h)
        
        roi = display_full[self.pan_y:end_y, self.pan_x:end_x]
        
        # 3. Redimensionar para o tamanho original da janela (Upscale)
        # Usamos INTER_NEAREST para manter os pixels "quadrados" quando der zoom (pixel art style)
        # isso ajuda a ver exatamente onde você está pintando.
        view_img = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 4. Adicionar HUD (Texto)
        info = f"Zoom: {self.zoom_level:.1f}x | Pan: ({self.pan_x},{self.pan_y})"
        cv2.putText(view_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(view_img, f"Img: {self.current_idx+1}/{len(self.image_paths)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(view_img, f"Pincel: {self.brush_size} | WASD move", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return view_img

    def save_current_mask(self):
        original_name = os.path.basename(self.image_paths[self.current_idx])
        filename = os.path.splitext(original_name)[0] + ".png"
        save_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(save_path, self.mask)
        print(f"Salvo: {save_path}")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) # Permite redimensionar janela
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("--- CONTROLES AVANÇADOS ---")
        print("Mouse Scroll: Zoom In / Zoom Out")
        print("Teclas W, A, S, D: Mover a imagem (Pan) quando estiver com Zoom")
        print("Botão ESQUERDO: Pintar | Botão DIREITO: Apagar")
        print("P: Salvar | N: Próxima Imagem | C: Limpar | ESC: Sair")
        print("---------------------------")

        while True:
            if self.img is None:
                path = self.image_paths[self.current_idx]
                self.img = cv2.imread(path)
                
                # Carrega máscara existente ou cria nova
                original_name = os.path.basename(path)
                mask_name = os.path.splitext(original_name)[0] + ".png"
                existing_mask_path = os.path.join(self.output_dir, mask_name)
                
                if os.path.exists(existing_mask_path):
                    self.mask = cv2.imread(existing_mask_path, cv2.IMREAD_GRAYSCALE)
                    _, self.mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)
                    # Resize se tamanho diferir
                    if self.mask.shape != self.img.shape[:2]:
                         self.mask = cv2.resize(self.mask, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)

                # Resetar zoom ao mudar de imagem
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0

            # Exibe
            try:
                display = self.get_view_image()
                cv2.imshow(self.window_name, display)
            except Exception as e:
                print(f"Erro display: {e}")

            key = cv2.waitKey(20) & 0xFF

            # Movimentação (PAN) com WASD (move 50 pixels por vez, ou menos se tiver muito zoom)
            step = int(50 / self.zoom_level)
            if key == ord('w'): self.pan_y -= step
            elif key == ord('s'): self.pan_y += step
            elif key == ord('a'): self.pan_x -= step
            elif key == ord('d'): self.pan_x += step
            
            # Limites
            self.update_pan_limits()

            # Outros comandos
            if key == 27: break # ESC
            elif key == ord(']'): self.brush_size += 1
            elif key == ord('['): self.brush_size = max(1, self.brush_size - 1)
            elif key == ord('c'): self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            elif key == ord('p'): self.save_current_mask()
            elif key == ord('n'): 
                self.save_current_mask()
                if self.current_idx < len(self.image_paths) - 1:
                    self.current_idx += 1
                    self.img = None
            elif key == ord('b') and self.current_idx > 0:
                self.current_idx -= 1
                self.img = None

        cv2.destroyAllWindows()

if __name__ == "__main__":
    labeller = ImageLabelerZoom(input_dir=PASTA_IMAGENS, output_dir=PASTA_MASKS)
    labeller.run()