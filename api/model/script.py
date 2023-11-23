import torch
import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os

from model.network import Generator


sd_path = 'weights/save_weights.pth'



class DetectInpaint:
    def __init__(self, image:Image, mask:Image=None, task:str=None, use_cuda_if_available:bool=True) -> None:
        self.image = image
        if mask is not None:
            self.mask = mask
        elif task == 'detect' or task is None:
            self.model = YOLO('weights/best.pt')
            self.mask = self.detect_mask(img=self.image)
        elif task == 'avito':
            self.mask = self.mask_of_templates(img=self.image, name=task)
        else:
            raise ValueError ('Not found task')
        self.use_cuda_if_available = use_cuda_if_available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() 
            and use_cuda_if_available else 'cpu'
        )
        self.generator = Generator(checkpoint=sd_path, return_flow=True).to(self.device)
        
    
    def run(self):
        image = T.ToTensor()(self.image).to(self.device)
        mask = T.ToTensor()(self.mask).to(self.device)
        
        output = self.generator.infer(image, mask)

        image = Image.fromarray(output)
        
        return image
    
    
    def detect_mask(self, img:Image):
        img = img.copy()
        predict = self.model([img], conf=0.5)[0]  
        mask = Image.new('RGBA', tuple(list(predict.boxes.orig_shape)[::-1]), (0, 0, 0))
        mask1 = ImageDraw.Draw(mask)
        for wm_xyxy in predict.boxes.xyxy:
            xyxy = list(map(int, wm_xyxy))
            mask1.rectangle([(xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])], fill='white')
        return mask
        
    
    def find_mask_in_list(self, substring:str, string_list:list[str]) -> str:
        for string in string_list:
            if substring in string:
                return string
        raise ValueError ("Маска не найдена")


    def mask_of_templates(self, img:Image, name):
        directory = 'masks'
        files = os.listdir(directory)
        name_file = self.find_mask_in_list(substring=name, string_list=files)
        mask = Image.open(directory + '/' + name_file)
        
        if 'avito1' in name_file:
            mask_pil = Image.new('RGB', img.size, 'black')
            mask_pil.paste(mask, (mask_pil.size[0]-mask.size[0], mask_pil.size[1]-mask.size[1]))
        else:
            raise ValueError
        return mask_pil