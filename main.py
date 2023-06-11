from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch
from PIL import Image
import os
import uvicorn
import torchvision.transforms as T

app = FastAPI()
device = torch.device('cpu')
model = "pretrained/states_pt_places2.pth"
generator_state_dict = torch.load(model, map_location=device)['G']
if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
    from model.networks import Generator
else:
    from model.networks_tf import Generator

generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
generator_state_dict = torch.load(model, map_location=device)['G']
generator.load_state_dict(generator_state_dict, strict=True)


@app.post("/")
async def upload_image(image: UploadFile, mask: UploadFile):
    if image.filename == '':
        return {"message": "No image selected for uploading"}
    if mask.filename == '':
        return {"message": "No mask selected for uploading"}

    image_name = image.filename
    mask_name = mask.filename

    with open(f"{image_name}", "wb") as f:
        f.write(image.file.read())
    with open(f"{mask_name}", "wb") as f:
        f.write(mask.file.read())

    image = Image.open(image_name)
    mask = Image.open(mask_name)

    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h // grid * grid, :w // grid * grid].unsqueeze(0)
    mask = mask[0:1, :h // grid * grid, :w // grid * grid].unsqueeze(0)

    image = (image * 2 - 1.).to(device)  # map image values to [-1, 1] range
    mask = (mask > 0.5).to(dtype=torch.float32,
                           device=device)  # 1.: masked 0.: unmasked

    image_masked = image * (1. - mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x * mask],
                  dim=1)  # concatenate channels

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # complete image
    image_inpainted = image * (1. - mask) + x_stage2 * mask

    # save inpainted image
    output_name = "output.jpg"
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5)
    img_out = img_out.to(device=device, dtype=torch.uint8)
    img_out = Image.fromarray(img_out.numpy())
    img_out.save(output_name)

    os.remove(f'{image_name}')
    os.remove(f'{mask_name}')

    return FileResponse(f"{output_name}", media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
