import json
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import configparser
import torch.nn.functional as F

class ShiYiAIFlow:
    """
    ComfyUI图像生成节点 - 核心修正版
    1. 修正URL提取正则，完美适配Markdown格式
    2. 增加下载请求头，防止CDN拦截
    3. 保持批次内查重，批次间独立
    """
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    FUNCTION = "generate_images"
    OUTPUT_NODE = True
    CATEGORY = "image/ai_generation"
    
    RESOLUTION_MAP = {
        "1k": "1024x1024",
        "2k": "2048x2048", 
        "4k": "4096x4096"
    }
    
    ASPECT_RATIO_RESOLUTIONS = {
        "1:1": "1024x1024",
        "9:16": "768x1344",
        "16:9": "1344x768",
        "21:9": "1792x768",
        "2:3": "832x1248",
        "3:2": "1248x832",
        "3:4": "896x1152",
        "4:3": "1152x896",
        "4:5": "896x1120",
        "5:4": "1120x896"
    }
    
    @classmethod
    def load_config(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.ini")
        config = configparser.ConfigParser()
        default_config = {
            'api': {'default_url': 'Please enter your API proxy website', 'timeout': '180'},
            'model': {'default_model': 'gemini-2.5-flash-image', 'default_api_key': 'your-api-key-here'},
            'generation': {'default_resolution': '1k', 'default_aspect_ratio': 'Auto'}
        }
        if os.path.exists(config_path):
            try:
                config.read(config_path, encoding='utf-8')
            except Exception:
                config.read_dict(default_config)
        else:
            config.read_dict(default_config)
            try:
                with open(config_path, 'w', encoding='utf-8') as f: config.write(f)
            except Exception: pass
        return config
    
    @classmethod
    def INPUT_TYPES(cls):
        config = cls.load_config()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape", "placeholder": "请输入图像描述..."}),
                "api_base_url": ("STRING", {"default": config.get('api', 'default_url', fallback='Please enter your API proxy website')}),
                "model_type": ("STRING", {"default": config.get('model', 'default_model', fallback='gemini-2.5-flash-image')}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "aspect_ratio": (["Auto", "1:1", "9:16", "16:9", "21:9", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4"], {"default": config.get('generation', 'default_aspect_ratio', fallback='Auto')}),
                "resolution": (["1k", "2k", "4k"], {"default": config.get('generation', 'default_resolution', fallback='1k')})
            },
            "optional": {
                "api_key": ("STRING", {"default": config.get('model', 'default_api_key', fallback='your-api-key-here')}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 102400}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 8}),
                "input_image_1": ("IMAGE",), "input_image_2": ("IMAGE",), "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",), "input_image_5": ("IMAGE",),
            }
        }
    
    def tensor_to_base64(self, tensor: torch.Tensor) -> str:
        if tensor.dim() == 4: tensor = tensor[0]
        img_array = (torch.clamp(tensor, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def base64_to_tensor_single(self, b64_str: str) -> torch.Tensor:
        try:
            b64_str = b64_str.strip()
            if b64_str.startswith('data:image'):
                if ',' in b64_str: b64_str = b64_str.split(',', 1)[1]
            img_data = base64.b64decode(b64_str)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)
        except Exception as e:
            print(f"⚠️ Base64解码失败: {e}")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    
    def url_to_tensor_single(self, url: str) -> torch.Tensor:
        """下载URL图片，增加User-Agent防止CDN拦截"""
        try:
            # 严格清理URL
            clean_url = url.strip()
            
            # 设置浏览器头，防止poecdn等403 Forbidden
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            print(f"📥 下载图片: {clean_url}")
            response = requests.get(clean_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            print(f"✅ 下载成功: {img.size}")
            return torch.from_numpy(img_array).unsqueeze(0)
        except Exception as e:
            print(f"⚠️ 下载失败 [{url}]: {str(e)}")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    
    def convert_images_to_tensor(self, base64_images: List[str], image_urls: List[str]) -> torch.Tensor:
        """统一转换为Tensor并自动对齐维度"""
        all_tensors = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            if base64_images:
                futures = [executor.submit(self.base64_to_tensor_single, b64) for b64 in base64_images]
                for f in futures: all_tensors.append(f.result())
            
            if image_urls:
                futures = [executor.submit(self.url_to_tensor_single, url) for url in image_urls]
                for f in futures: all_tensors.append(f.result())
        
        if not all_tensors:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # 维度对齐逻辑：以第一个有效大图为准
        target_h, target_w = 1024, 1024
        valid_found = False
        for t in all_tensors:
            if t.shape[1] > 64 and t.shape[2] > 64:
                target_h, target_w = t.shape[1], t.shape[2]
                valid_found = True
                break
        
        if not valid_found and all_tensors:
             target_h, target_w = all_tensors[0].shape[1], all_tensors[0].shape[2]

        final_tensors = []
        for t in all_tensors:
            if t.shape[1] != target_h or t.shape[2] != target_w:
                t = t.permute(0, 3, 1, 2)
                t = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
                t = t.permute(0, 2, 3, 1)
            final_tensors.append(t)

        return torch.cat(final_tensors, dim=0)

    def create_request_data(self, prompt, seed, aspect_ratio, resolution, top_p, input_images):
        final_prompt = f"{prompt}, detailed, high quality" if seed != -1 else prompt
        parts = [{"text": final_prompt}]
        
        if input_images:
            for img in input_images:
                if img is not None:
                    parts.append({"inlineData": {"mimeType": "image/png", "data": self.tensor_to_base64(img)}})
        
        size = self.ASPECT_RATIO_RESOLUTIONS.get(aspect_ratio, "1024x1024") if aspect_ratio != "Auto" and resolution == "1k" else self.RESOLUTION_MAP.get(resolution, "1024x1024")
        
        config = {"temperature": 0.8, "topP": top_p, "maxOutputTokens": 8192, "size": size, "num_images": 1}
        if aspect_ratio != "Auto": config["aspectRatio"] = aspect_ratio
        if seed != -1: config["seed"] = seed
        
        return {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}

    def send_request(self, api_key, request_data, model_type, api_base_url, timeout=180):
        if "generativelanguage.googleapis.com" in api_base_url:
            url = f"{api_base_url.rstrip('/')}/v1beta/models/{model_type}:generateContent?key={api_key}"
            headers = {'Content-Type': 'application/json'}
        elif "openai.com" in api_base_url:
            url = f"{api_base_url.rstrip('/')}/v1/images/generations"
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
        else:
            url = f"{api_base_url.rstrip('/')}/v1beta/models/{model_type}:generateContent"
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}' if api_key else ''}
            if api_key and 'Bearer' not in api_key:
                url += f"&key={api_key}" if '?' in url else f"?key={api_key}"
        
        headers['User-Agent'] = 'Shi-Yi-AI-Flow/1.0'
        
        try:
            print(f"🔗 请求: {url.split('?')[0]}")
            response = requests.post(url, json=request_data, headers=headers, timeout=timeout)
            if response.status_code != 200:
                raise Exception(f"API错误 {response.status_code}: {response.text[:200]}")
            return response.json()
        except Exception as e:
            raise Exception(str(e))

    def extract_kuai_host_urls(self, text: str) -> List[str]:
        """
        修正后的提取逻辑：
        您提供的响应中，URL格式为：(https://pfst...13468?w=1024&h=1024)
        Markdown格式通常为：[alt](url) 或 ![alt](url)
        
        我们使用简单的正则提取：'https://...' 直到遇到 括号、引号或空白
        """
        urls = []
        
        # 核心正则：匹配 http 或 https 开头，直到遇到 ), space, ", '
        # 这样可以准确地把 Markdown 的闭合括号 ')' 排除在外
        pattern = r'(https?://[^\s\)"\'<>]+)'
        
        found_urls = re.findall(pattern, text)
        
        cleaned_urls = []
        seen_in_text = set() # 仅在本文本内去重
        
        for url in found_urls:
            # 清理残留的标点（保险起见）
            url = url.rstrip('.,;)]}')
            
            if 'pfst.cf2.poecdn.net' in url or url.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # 检查是否已有宽高等参数
                if '?' not in url and 'pfst.cf2.poecdn.net' in url:
                     url += '?w=1024&h=1024'
                
                if url not in seen_in_text:
                    cleaned_urls.append(url)
                    seen_in_text.add(url)
                    
        return cleaned_urls

    def extract_content(self, response_data: Dict) -> Tuple[List[str], List[str], str]:
        base64_images = []
        image_urls = []
        text_content = ""
        seen_urls = set() # 批次内去重 (Requirement 1)
        
        try:
            candidates = response_data.get('candidates', [])
            for candidate in candidates:
                for part in candidate.get('content', {}).get('parts', []):
                    if 'text' in part:
                        text = part['text']
                        text_content += text
                        
                        # 使用新逻辑提取
                        urls = self.extract_kuai_host_urls(text)
                        for url in urls:
                            if url not in seen_urls:
                                image_urls.append(url)
                                seen_urls.add(url)
                                print(f"    🎯 提取URL: {url}")
                                
                    elif 'inlineData' in part:
                        data = part['inlineData']['data']
                        if len(data) > 100: base64_images.append(data)

            # 兜底逻辑：如果上面的parse没找到，尝试全局搜索
            if not image_urls and not base64_images:
                response_str = json.dumps(response_data)
                # 同样的全局正则搜索
                raw_urls = re.findall(r'(https?://[^\s\)"\'<>]+)', response_str)
                for url in raw_urls:
                    url = url.rstrip('\\"') # 清理json转义符
                    if 'pfst.cf2.poecdn.net' in url and url not in seen_urls:
                         image_urls.append(url)
                         seen_urls.add(url)
                         print(f"    🎯 兜底提取URL: {url}")

            print(f"📊 解析结果: URL图片 {len(image_urls)} 张")
            
        except Exception as e:
            print(f"⚠️ 解析错误: {e}")
        
        return base64_images, image_urls, text_content

    def generate_single_image(self, args):
        (i, current_seed, api_key, prompt, model_type, aspect_ratio, resolution, top_p, input_images, api_base_url, timeout) = args
        try:
            req = self.create_request_data(prompt, current_seed, aspect_ratio, resolution, top_p, input_images)
            res = self.send_request(api_key, req, model_type, api_base_url, timeout)
            b64, urls, txt = self.extract_content(res)
            return {'index': i, 'success': True, 'base64_images': b64, 'image_urls': urls, 'text': txt}
        except Exception as e:
            return {'index': i, 'success': False, 'error': str(e)}

    def generate_images(self, prompt, api_base_url, model_type, batch_size, aspect_ratio, resolution, 
                       api_key=None, seed=-1, top_p=0.95, max_workers=4, 
                       input_image_1=None, input_image_2=None, input_image_3=None, input_image_4=None, input_image_5=None):
        
        config = self.load_config()
        if not api_key or api_key == "your-api-key-here":
            api_key = config.get('model', 'default_api_key', fallback='your-api-key-here')
            
        input_images = [x for x in [input_image_1, input_image_2, input_image_3, input_image_4, input_image_5] if x is not None]
        base_seed = random.randint(0, 102400) if seed == -1 else seed
        
        all_b64, all_urls, all_texts = [], [], []
        
        print(f"\n{'='*40}\n🚀 开始生成 (Batch: {batch_size})\n{'='*40}")
        
        tasks = [(i, base_seed+i if seed!=-1 else -1, api_key, prompt, model_type, aspect_ratio, resolution, top_p, input_images, api_base_url, 180) for i in range(batch_size)]
        
        # 串行还是并行
        if batch_size > 1 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(batch_size, max_workers)) as exe:
                futures = [exe.submit(self.generate_single_image, t) for t in tasks]
                for f in as_completed(futures):
                    r = f.result()
                    if r['success']:
                        all_b64.extend(r['base64_images'])
                        all_urls.extend(r['image_urls'])
                        if r['text']: all_texts.append(f"[#{r['index']+1}] {r['text']}")
                    else:
                        all_texts.append(f"[#{r['index']+1}] ❌ {r['error']}")
        else:
            for t in tasks:
                r = self.generate_single_image(t)
                if r['success']:
                    all_b64.extend(r['base64_images'])
                    all_urls.extend(r['image_urls'])
                    if r['text']: all_texts.append(f"[#{r['index']+1}] {r['text']}")
                else:
                    all_texts.append(f"[#{r['index']+1}] ❌ {r['error']}")
        
        # 最终转换
        final_tensor = self.convert_images_to_tensor(all_b64, all_urls)
        print(f"\n✅ 任务完成: 获取 {final_tensor.shape[0]} 张图片\n")
        
        return (final_tensor, "\n".join(all_texts))

NODE_CLASS_MAPPINGS = {"ShiYiAIFlow": ShiYiAIFlow}
NODE_DISPLAY_NAME_MAPPINGS = {"ShiYiAIFlow": "Shi-Yi ai流🌐"}
