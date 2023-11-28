import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler

'''WEIGHTS_DIR="stable_diffusion_weights/pair123987/1000"
positive_prompt="kajal123987 person with pavan123098 person"
negative_prompts=""
'''
def sd_inference(WEIGHTS_DIR,positive_prompt,negative_prompts):
    model_path =  natsorted(glob(WEIGHTS_DIR + os.sep + "*"))[-1]
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = None
    num_samples = 6
    guidance_scale = 8
    num_inference_steps = 50
    height = 512
    width = 512
    prompt=positive_prompt
    negative_prompt=negative_prompts
    with autocast("cuda"), torch.inference_mode():
        images = pipe(
                prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g_cuda
                ).images
    #for i in range(len(images)):
    #    images[i].save(f"{i}.png")
    return images


'''if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=9093,debug=True)
    sd_inference()
'''