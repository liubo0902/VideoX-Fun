import torch
from diffusers import QwenImageControlNetPipeline, QwenImageControlNetModel
from PIL import Image

base_model = "Qwen/Qwen-Image"
controlnet_model = "InstantX/Qwen-Image-ControlNet-Union"

controlnet = QwenImageControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = QwenImageControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")
control_image = Image.open("asset/pose.jpg")
prompt = "画面中央是一位年轻女孩，她拥有一头令人印象深刻的亮紫色长发，发丝在海风中轻盈飘扬，营造出动感而唯美的效果。她的长发两侧各扎着黑色蝴蝶结发饰，增添了几分可爱与俏皮感。女孩身穿一袭纯白色无袖连衣裙，裙摆轻盈飘逸，与她清新的气质完美契合。她的妆容精致自然，淡粉色的唇妆和温柔的眼神流露出恬静优雅的气质。她单手叉腰，姿态自信从容，目光直视镜头，展现出既甜美又不失个性的魅力。背景是一片开阔的海景，湛蓝的海水在阳光照射下波光粼粼，闪烁着钻石般的光芒。天空呈现出清澈的蔚蓝色，点缀着几朵洁白的云朵，营造出晴朗明媚的夏日氛围。画面前景右下角可见粉紫色的小花丛和绿色植物，为整体构图增添了自然生机和色彩层次。整张照片色调明亮清新，紫色头发与白色裙装、蓝色海天形成鲜明而和谐的色彩对比。"
negative_prompt = ""
seed = 42
controlnet_conditioning_scale = 1.0
cfg_scale = 4.0
num_inference_steps = 30
save_filename = 'pos_cn.png'

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    control_image=control_image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=control_image.size[0],
    height=control_image.size[1],
    num_inference_steps=num_inference_steps,
    true_cfg_scale=cfg_scale,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]

image.save(f"{save_filename}")
