import math
import random

import cv2
import gradio as gr
import numpy as np
import PIL
import spaces
import torch
from diffusers.models import ControlNetModel
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from PIL import Image

from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from style_template import styles

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"

# download checkpoints
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="./checkpoints",
)
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")

# Load face encoder
app = FaceAnalysis(name="antelopev2", root="./", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = "./checkpoints/ip-adapter.bin"
controlnet_path = "./checkpoints/ControlNetModel"

# Load pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base_model_path = "wangqixun/YamerMIX_v8"

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
)
pipe.cuda()
pipe.load_ip_adapter_instantid(face_adapter)
pipe.image_proj_model.to("cuda")
pipe.unet.to("cuda")


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def remove_tips():
    return gr.update(visible=False)


def get_example():
    case = [
        [
            "./examples/yann-lecun_resize.jpg",
            "a man",
            "Snow",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/musk_resize.jpeg",
            "a man",
            "Mars",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/sam_resize.png",
            "a man",
            "Jungle",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
        ],
        [
            "./examples/schmidhuber_resize.png",
            "a man",
            "Neon",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/kaifu_resize.png",
            "a man",
            "Vibrant Color",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
    ]
    return case


def run_for_examples(face_file, prompt, style, negative_prompt):
    return generate_image(face_file, None, prompt, negative_prompt, style, True, 30, 0.8, 0.8, 5, 42)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative


def check_input_image(face_image):
    if face_image is None:
        raise gr.Error("Cannot find any input face image! Please upload the face image")


@spaces.GPU
def generate_image(
    face_image_path,
    pose_image_path,
    prompt,
    negative_prompt,
    style_name,
    enhance_face_region,
    num_steps,
    identitynet_strength_ratio,
    adapter_strength_ratio,
    guidance_scale,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    if prompt is None:
        prompt = "a person"

    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    face_image = load_image(face_image_path)
    face_image = resize_img(face_image)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Extract face features
    face_info = app.get(face_image_cv2)

    if len(face_info) == 0:
        raise gr.Error("Cannot find any face in the image! Please upload another person image")

    face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1])[
        -1
    ]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

    if pose_image_path is not None:
        pose_image = load_image(pose_image_path)
        pose_image = resize_img(pose_image)
        pose_image_cv2 = convert_from_image_to_cv2(pose_image)

        face_info = app.get(pose_image_cv2)

        if len(face_info) == 0:
            raise gr.Error("Cannot find any face in the reference image! Please upload another person image")

        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info["kps"])

        width, height = face_kps.size

    if enhance_face_region:
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Start inference...")
    print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        control_mask=control_mask,
        controlnet_conditioning_scale=float(identitynet_strength_ratio),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images

    return images[0], gr.update(visible=True)


### Description
title = r"""
<h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

How to use:<br>
1. Upload a person image. For multiple person images, we will only detect the biggest face. Make sure face is not too small and not significantly blocked or blurred.
2. (Optionally) upload another person image as reference pose. If not uploaded, we will use the first person image to extract landmarks. If you use a cropped face at step1, it is recommeneded to upload it to extract a new pose.
3. Enter a text prompt as done in normal text-to-image models.
4. Click the <b>Submit</b> button to start customizing.
5. Share your customizd photo with your friends, enjoy😊!
"""

article = r"""
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{wang2024instantid,
  title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
  author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2401.07519},
  year={2024}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out at <b>haofanwang.ai@gmail.com</b>.
"""

tips = r"""
### Usage tips of InstantID
1. If you're unsatisfied with the similarity, increase the weight of controlnet_conditioning_scale (IdentityNet) and ip_adapter_scale (Adapter).
2. If the generated image is over-saturated, decrease the ip_adapter_scale. If not work, decrease controlnet_conditioning_scale.
3. If text control is not as expected, decrease ip_adapter_scale.
4. Find a good base model always makes a difference.
"""

css = """
.gradio-container {width: 85% !important}
"""
with gr.Blocks(css=css) as demo:
    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            # upload face image
            face_file = gr.Image(label="Upload a photo of your face", type="filepath")

            # optional: upload a reference pose image
            pose_file = gr.Image(label="Upload a reference pose image (optional)", type="filepath")

            # prompt
            prompt = gr.Textbox(
                label="Prompt",
                info="Give simple prompt is enough to achieve good face fedility",
                placeholder="A photo of a person",
                value="",
            )

            submit = gr.Button("Submit", variant="primary")

            style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)

            # strength
            identitynet_strength_ratio = gr.Slider(
                label="IdentityNet strength (for fedility)",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=0.80,
            )
            adapter_strength_ratio = gr.Slider(
                label="Image adapter strength (for detail)",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=0.80,
            )

            with gr.Accordion(open=False, label="Advanced Options"):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="low quality",
                    value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                )
                num_steps = gr.Slider(
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=30,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

        with gr.Column():
            output_image = gr.Image(label="Generated Image")
            usage_tips = gr.Markdown(label="Usage tips of InstantID", value=tips, visible=False)

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,
            queue=False,
            api_name=False,
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=check_input_image,
            inputs=face_file,
            queue=False,
            api_name=False,
        ).success(
            fn=generate_image,
            inputs=[
                face_file,
                pose_file,
                prompt,
                negative_prompt,
                style,
                enhance_face_region,
                num_steps,
                identitynet_strength_ratio,
                adapter_strength_ratio,
                guidance_scale,
                seed,
            ],
            outputs=[output_image, usage_tips],
        )

    gr.Examples(
        examples=get_example(),
        inputs=[face_file, prompt, style, negative_prompt],
        outputs=[output_image, usage_tips],
        fn=run_for_examples,
        cache_examples=True,
    )

    gr.Markdown(article)

demo.queue(api_open=False)
demo.launch()
